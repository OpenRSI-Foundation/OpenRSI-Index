from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.rsi_loop import (
    AGENT_SHELL,
    RSILoopAgentAdapter,
    validate_agent_config,
)
from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    RunGPUPlan,
)
from rsi_loop.harness.agent import create_agent, list_agent_classes
from rsi_loop.harness.agent.base import NODE_BIN, STOP_HOOK_REASON
from rsi_loop.harness.config import RSILoopConfig, load_config
from tests.factories import make_run_plan
from tests.fakes import register_fake_driven_agent


class RecordingAgentRuntime:
    def __init__(self) -> None:
        self.copies: list[tuple[str, Path, PurePosixPath]] = []
        self.executions: list[dict[str, object]] = []
        self.result = AgentRunResult(exit_code=0)

    def copy_to(
        self, container: ContainerRef, source: Path, target: PurePosixPath
    ) -> None:
        self.copies.append((container.container_id, source, target))

    def exec(
        self,
        container: ContainerRef,
        command: str | tuple[str, ...] | list[str],
        *,
        timeout_seconds: float | None = None,
        user: str | None = None,
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
        output_redact_values: tuple[str, ...] = (),
        output_callback: Callable[[str], None] | None = None,
    ) -> AgentRunResult:
        execution: dict[str, object] = {
            "container": container.container_id,
            "command": command,
            "timeout_seconds": timeout_seconds,
            "user": user,
            "environment": environment,
        }
        if output_path is not None:
            execution["output_path"] = output_path
        if output_redact_values:
            execution["output_redact_values"] = output_redact_values
        if output_callback is not None:
            execution["output_callback"] = output_callback
        self.executions.append(execution)
        return self.result


def test_available_agents_tracks_rsi_loop_registry() -> None:
    expected = tuple(sorted(cls.name for cls in list_agent_classes()))

    assert RSILoopAgentAdapter(RSILoopConfig()).available_agents() == expected


def test_prepare_writes_archive_free_workdir_prompt_and_resolves_agent_env(
    tmp_path: Path,
) -> None:
    prompt_path = (tmp_path / "rsi-agent-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    config = RSILoopConfig(
        http_proxy="http://proxy.internal:8080",
        agent_api_key="runtime-secret",
        agent_model="gpt-test",
        agent_extra_env={"EXTRA": "value"},
    )

    prepared = RSILoopAgentAdapter(config).prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path)
    )

    prompt = prompt_path.read_text()
    assert "entire current WORKDIR" in prompt
    assert "best valid primary score wins" in prompt
    assert "rsi-submit" in prompt
    assert "repeatedly" in prompt
    assert "`rsi-submit --list` shows previous submissions" in prompt
    assert "`rsi-submit --help` shows local usage" in prompt
    assert "/run/rsi-harness/feedback/agent-N.log" in prompt
    assert "complete Judge stdout and stderr" in prompt
    assert "package" not in prompt.lower()
    assert "upload" not in prompt.lower()
    assert "runtime-secret" not in prompt
    assert prepared.command[:2] == ("/bin/bash", "-c")
    assert "/tmp/rsi-agent-prompt.md" in prepared.command[2]
    assert "gpt-test" in prepared.command[2]
    assert dict(prepared.environment) == {
        "CODEX_API_KEY": "runtime-secret",
        "CODEX_MODEL": "gpt-test",
        "EXTRA": "value",
        "HTTP_PROXY": "http://proxy.internal:8080",
        "OPENAI_API_KEY": "runtime-secret",
        "http_proxy": "http://proxy.internal:8080",
    }


@pytest.mark.parametrize(
    ("max_submissions", "expected_budget"),
    (
        (2, "You may submit to the Judge at most 2 times during this run."),
        (None, "Judge submissions are unlimited during this run."),
    ),
)
def test_prepare_prompt_states_the_submission_budget(
    tmp_path: Path,
    max_submissions: int | None,
    expected_budget: str,
) -> None:
    """Omitting the budget can make an Agent waste its final Judge attempt."""
    prompt_path = (tmp_path / "submission-budget-prompt.md").resolve()

    RSILoopAgentAdapter(RSILoopConfig()).prepare(
        AgentPrepareRequest(
            run_plan=make_run_plan(tmp_path),
            prompt_path=prompt_path,
            max_submissions=max_submissions,
        )
    )

    assert expected_budget in prompt_path.read_text()


@pytest.mark.parametrize(
    ("mode", "judge_devices", "expected_guidance"),
    (
        (
            JudgeGPUMode.FREEZE_ONLY,
            (),
            "Work is paused during evaluation; GPU release is not required.",
        ),
        (
            JudgeGPUMode.DISJOINT,
            (GPUDevice(index=2, uuid="GPU-c", name="H100"),),
            "Judge uses 1 separate GPU; Work GPU release is not required.",
        ),
        (
            JudgeGPUMode.RELEASE_ALL,
            (GPUDevice(index=0, uuid="GPU-a", name="H100"),),
            "Every Work GPU process must exit before rsi-submit. A rejected "
            "preflight does not consume a submission.",
        ),
    ),
)
def test_prepare_gpu_mode_prompt_guidance_exposes_no_device_or_control_values(
    tmp_path: Path,
    mode: JudgeGPUMode,
    judge_devices: tuple[GPUDevice, ...],
    expected_guidance: str,
) -> None:
    """Wrong mode text could cause an Agent to release GPUs unnecessarily."""
    plan = make_run_plan(tmp_path).model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(
                    devices=(
                        GPUDevice(index=0, uuid="GPU-a", name="H100"),
                        GPUDevice(index=1, uuid="GPU-b", name="H100"),
                        GPUDevice(index=2, uuid="GPU-c", name="H100"),
                    )
                ),
                work=GPUAllocation(
                    devices=(
                        GPUDevice(index=0, uuid="GPU-a", name="H100"),
                        GPUDevice(index=1, uuid="GPU-b", name="H100"),
                    )
                ),
                judge=GPUAllocation(devices=judge_devices),
                judge_mode=mode,
            )
        }
    )
    config = RSILoopConfig(agent_api_key="provider-secret")
    prompt_path = (tmp_path / f"{mode}-prompt.md").resolve()

    RSILoopAgentAdapter(config).prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path)
    )

    prompt = prompt_path.read_text()
    assert expected_guidance in prompt
    for forbidden in (
        "GPU-a",
        "GPU-b",
        "GPU-c",
        "provider-secret",
        "RSI_JUDGE_URL",
        "RSI_TOKEN",
    ):
        assert forbidden not in prompt


def test_public_configuration_uses_rsi_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An RSI run must not require the retired legacy public namespace."""
    for key in (
        "HTTP_PROXY",
        "http_proxy",
        "RSI_AGENT_API_KEY",
        "RSI_AGENT_API_BASE_URL",
        "RSI_AGENT_MODEL",
        "RSI_AGENT_EXTRA_ENV",
        "RSI_CLAUDE_CACHE_OPT",
        "RSI_HTTP_PROXY",
        "RSI_NODEJS_MIRROR_URL",
        "RSI_NPM_REGISTRY_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RSI_AGENT_API_KEY", "rsi-key")
    monkeypatch.setenv("RSI_AGENT_API_BASE_URL", "https://agent.rsi.test/v1")
    monkeypatch.setenv("RSI_AGENT_MODEL", "rsi-model")
    monkeypatch.setenv("RSI_AGENT_EXTRA_ENV", "EXTRA_ONE=one,EXTRA_TWO=two")
    monkeypatch.setenv("RSI_CLAUDE_CACHE_OPT", "1")
    monkeypatch.setenv("RSI_HTTP_PROXY", "http://proxy.rsi.test:8080")
    monkeypatch.setenv("RSI_NODEJS_MIRROR_URL", "https://node.rsi.test")
    monkeypatch.setenv("RSI_NPM_REGISTRY_URL", "https://npm.rsi.test")

    config = load_config()

    assert config.agent_api_key == "rsi-key"
    assert config.agent_api_base_url == "https://agent.rsi.test/v1"
    assert config.agent_model == "rsi-model"
    assert config.agent_extra_env == {"EXTRA_ONE": "one", "EXTRA_TWO": "two"}
    assert config.claude_cache_opt is True
    assert config.http_proxy == "http://proxy.rsi.test:8080"
    assert config.nodejs_mirror_url == "https://node.rsi.test"
    assert config.npm_registry_url == "https://npm.rsi.test"


@pytest.mark.parametrize("effort", ("xhigh", "max", "ultra"))
def test_prepare_maps_generic_reasoning_effort_to_codex_cli_override(
    tmp_path: Path, effort: str
) -> None:
    prompt_path = (tmp_path / "reasoning-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"model": "gpt-test", "reasoning_effort": effort}
                    )
                }
            )
        }
    )

    prepared = RSILoopAgentAdapter(RSILoopConfig()).prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path)
    )

    assert prepared.command == (
        "/bin/bash",
        "-c",
        f"codex exec -c 'model_reasoning_effort=\"{effort}\"' "
        '-c web_search="disabled" -c allow_login_shell=false '
        "-c features.shell_snapshot=false --model gpt-test "
        "--dangerously-bypass-approvals-and-sandbox "
        '"$(cat /tmp/rsi-agent-prompt.md)"',
    )


@pytest.mark.parametrize("via_extra_env", (False, True))
def test_codex_command_names_the_configured_endpoint_as_its_provider(
    tmp_path: Path, via_extra_env: bool
) -> None:
    """The pinned Codex ignores OPENAI_BASE_URL; its command names a provider.

    The URL is read from the variable when the command runs, so the logged
    command does not carry it. Without a configured endpoint Codex keeps its
    default provider.
    """
    url = "https://agent.rsi.test/v1"
    config = (
        RSILoopConfig(agent_extra_env={"OPENAI_BASE_URL": url})
        if via_extra_env
        else RSILoopConfig(agent_api_base_url=url)
    )
    command = create_agent("codex", config).format_run_cmd(
        "/tmp/prompt.md", model="gpt-test"
    )
    default = create_agent("codex", RSILoopConfig()).format_run_cmd(
        "/tmp/prompt.md", model="gpt-test"
    )

    assert url not in command
    assert "model_provider" not in default
    argv = subprocess.run(
        ["bash", "-c", 'codex() { printf "%s\\n" "$@"; }; ' + command],
        env={**os.environ, "OPENAI_BASE_URL": url},
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    overrides = [value for flag, value in zip(argv, argv[1:]) if flag == "-c"]
    assert sorted(overrides) == [
        "allow_login_shell=false",
        "features.shell_snapshot=false",
        "model_provider=rsi_proxy",
        f'model_providers.rsi_proxy.base_url="{url}"',
        "model_providers.rsi_proxy.env_key=OPENAI_API_KEY",
        "model_providers.rsi_proxy.name=rsi_proxy",
        "model_providers.rsi_proxy.wire_api=responses",
    ]


@pytest.mark.parametrize(
    ("resume", "expected_prefix"),
    (
        (False, "claude --effort max -p --output-format stream-json"),
        (True, 'claude --effort max --continue -p "Continue working."'),
    ),
)
def test_prepare_maps_generic_reasoning_effort_to_claude_cli_flag(
    tmp_path: Path, resume: bool, expected_prefix: str
) -> None:
    prompt_path = (tmp_path / "claude-reasoning-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "name": "claude-code",
                            "model": "claude-opus-5",
                            "reasoning_effort": "max",
                        }
                    )
                }
            )
        }
    )

    prepared = RSILoopAgentAdapter(RSILoopConfig()).prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=prompt_path,
            resume=resume,
        )
    )

    assert prepared.command[:2] == ("/bin/bash", "-c")
    assert prepared.command[2].startswith(expected_prefix)
    assert "--model claude-opus-5" in prepared.command[2]
    assert "--reasoning-effort" not in prepared.command[2]


def test_claude_agent_environment_marks_work_container_as_sandbox() -> None:
    from rsi_harness.integrations.rsi_loop import (
        rsi_loop_agent_environment,
        rsi_loop_runtime_secret_values,
    )
    from rsi_loop.harness.agent import create_agent

    config = RSILoopConfig()
    environment = rsi_loop_agent_environment(
        config, create_agent("claude-code", config), "claude-opus-5"
    )

    # Task images without a USER run the Agent as root, where Claude Code
    # refuses --dangerously-skip-permissions unless IS_SANDBOX is exactly "1".
    assert environment["IS_SANDBOX"] == "1"
    # The marker must not join the exact-value redaction set (it would
    # redact every "1" in the trajectory).
    assert "1" not in rsi_loop_runtime_secret_values(config)
    codex_environment = rsi_loop_agent_environment(
        config, create_agent("codex", config), None
    )
    assert "IS_SANDBOX" not in codex_environment


def test_claude_stop_hook_cap_is_disabled_without_redacting_zeroes() -> None:
    from rsi_harness.integrations.rsi_loop import (
        rsi_loop_agent_environment,
        rsi_loop_runtime_secret_values,
    )
    from rsi_harness.runtime.redaction import redact_exact_values
    from rsi_loop.harness.agent import create_agent

    config = RSILoopConfig(agent_api_key="provider-secret")
    environment = rsi_loop_agent_environment(
        config, create_agent("claude-code", config), None
    )

    assert environment["CLAUDE_CODE_STOP_HOOK_BLOCK_CAP"] == "0"
    assert (
        redact_exact_values(
            "step 10, loss 0.01, provider-secret",
            rsi_loop_runtime_secret_values(config),
        )
        == "step 10, loss 0.01, [REDACTED]"
    )
    codex_environment = rsi_loop_agent_environment(
        config, create_agent("codex", config), None
    )
    assert "CLAUDE_CODE_STOP_HOOK_BLOCK_CAP" not in codex_environment


@pytest.mark.parametrize(
    ("name", "value", "is_credential"),
    (
        ("DISABLE_AUTOUPDATER", "1", False),
        ("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", "0", False),
        ("TOKENIZERS_PARALLELISM", "false", False),
        ("MAX_OUTPUT_TOKENS", "1000", False),
        ("TOKEN_COUNT", "10", False),
        ("API_KEY_FILE", "/tmp/key", False),
        ("HF_HOME", "/tmp/cache", False),
        ("MODEL_NAME", "some-model", False),
        ("CACHE_KEY", "shared-cache", False),
        ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1", False),
        ("REQUIRE_API_KEY", "1", False),
        ("EOS_TOKEN", "151643", False),
        ("CUSTOM_TOKEN", "ordinary-value", False),
        ("ANTHROPIC_AUTH_TOKEN", "provider-credential", True),
        ("OPENAI_API_KEY", "provider-credential", True),
        ("HF_TOKEN", "hub-credential", True),
        ("NPM_TOKEN", "npm-credential", True),
        ("PYPI_TOKEN", "pypi-credential", True),
        ("AWS_ACCESS_KEY_ID", "access-credential", True),
        ("AWS_SECRET_ACCESS_KEY", "aws-credential", True),
        ("AWS_SESSION_TOKEN", "session-credential", True),
        ("CLIENT_SECRET", "client-credential", True),
        ("CUSTOM_PASSWORD", "password-credential", True),
        ("CUSTOM_PRIVATE_KEY", "private-credential", True),
        ("HF_TOKEN", "1", True),
    ),
)
def test_extra_env_redaction_distinguishes_credentials_from_settings(
    name: str, value: str, is_credential: bool
) -> None:
    from rsi_harness.integrations.rsi_loop import rsi_loop_runtime_secret_values

    config = RSILoopConfig(agent_extra_env={name: value})

    assert (value in rsi_loop_runtime_secret_values(config)) is is_credential


def test_extra_env_custom_credential_names_are_explicitly_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rsi_harness.integrations.rsi_loop import rsi_loop_runtime_secret_values

    monkeypatch.setenv(
        "RSI_AGENT_EXTRA_ENV", "CUSTOM_VALUE=opaque-credential,DISABLE_AUTOUPDATER=1"
    )
    monkeypatch.setenv("RSI_AGENT_SECRET_ENV_NAMES", " CUSTOM_VALUE, , MISSING ")
    config = load_config()

    assert config.agent_secret_env_names == ("CUSTOM_VALUE", "MISSING")
    secrets = rsi_loop_runtime_secret_values(config)
    assert "opaque-credential" in secrets
    assert "1" not in secrets
    assert config.agent_extra_env["DISABLE_AUTOUPDATER"] == "1"


def test_extra_env_authenticated_urls_still_redact_embedded_credentials() -> None:
    from rsi_harness.integrations.rsi_loop import rsi_loop_runtime_secret_values

    endpoint = "https://proxy-user:p%40ssword@proxy.example:8443"
    config = RSILoopConfig(
        agent_extra_env={
            "HTTPS_PROXY": endpoint,
            "MODEL_ENDPOINT": "https://model.example/v1",
            "ORDINARY_TEXT": "http://[unfinished",
        }
    )

    assert rsi_loop_runtime_secret_values(config) == {
        endpoint,
        "proxy-user",
        "p%40ssword",
        "p@ssword",
    }


def test_explicit_secret_names_override_automatic_credential_classification() -> None:
    from rsi_harness.integrations.rsi_loop import rsi_loop_runtime_secret_values

    config = RSILoopConfig(
        agent_extra_env={"CUSTOM_TOKEN": "1", "USE_SECRET": "custom-credential"},
        agent_secret_env_names=("CUSTOM_TOKEN", "USE_SECRET"),
    )
    assert rsi_loop_runtime_secret_values(config) == {"1", "custom-credential"}


def test_extra_env_settings_preserve_streamed_trajectory_and_hide_credentials(
    tmp_path: Path,
) -> None:
    from rsi_harness.cluster.bluevela.runtime import _safe_output
    from rsi_harness.integrations.rsi_loop import rsi_loop_runtime_secret_values
    from rsi_harness.runtime.docker import _RedactedOutputWriter

    config = RSILoopConfig(
        agent_api_key="provider-credential",
        agent_extra_env={
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_STOP_HOOK_BLOCK_CAP": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/tmp/cache",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_TOKEN": "hub-credential",
            "CUSTOM_VALUE": "custom-credential",
        },
        agent_secret_env_names=("CUSTOM_VALUE",),
    )
    raw = (
        '{"type":"assistant","text":"Epoch 1/10 loss 0.01 false /tmp/cache '
        'provider-credential hub-credential custom-credential",'
        '"token_count":10,"enabled":true}\n'
    )
    expected = {
        "type": "assistant",
        "text": "Epoch 1/10 loss 0.01 false /tmp/cache "
        "[REDACTED] [REDACTED] [REDACTED]",
        "token_count": 10,
        "enabled": True,
    }
    secrets = rsi_loop_runtime_secret_values(config)
    output_path = tmp_path / "trajectory.jsonl"
    chunks: list[str] = []
    writer = _RedactedOutputWriter(output_path, tuple(secrets), chunks.append)
    encoded = raw.encode()
    for offset in range(0, len(encoded), 7):
        writer.append(encoded[offset : offset + 7])
    writer.finish()

    assert json.loads(output_path.read_text()) == expected
    assert "".join(chunks) == output_path.read_text()
    assert json.loads(_safe_output(raw, secrets)) == expected


def test_claude_prompt_is_read_from_stdin_and_never_expanded_into_argv(
    tmp_path: Path,
) -> None:
    """A task prompt in argv lets `pkill -f <task word>` kill the Agent itself."""
    prompt_path = (tmp_path / "claude-stdin-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"name": "claude-code", "model": "claude-opus-5"}
                    )
                }
            )
        }
    )

    prepared = RSILoopAgentAdapter(RSILoopConfig()).prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path, resume=False)
    )

    command = prepared.command[2]
    assert command.startswith("claude -p ")
    assert "$(cat" not in command
    assert " </tmp/rsi-agent-prompt.md" in command
    assert " --model claude-opus-5" in command


@pytest.mark.parametrize("effort", ("minimal", "ultra"))
def test_prepare_rejects_unsupported_claude_reasoning_effort_before_writing_prompt(
    tmp_path: Path, effort: str
) -> None:
    prompt_path = (tmp_path / "invalid-claude-reasoning-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "name": "claude-code",
                            "reasoning_effort": effort,
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(SetupError, match="Claude Code reasoning effort"):
        RSILoopAgentAdapter(RSILoopConfig()).prepare(
            AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path)
        )

    assert not prompt_path.exists()


@pytest.mark.parametrize("effort", ("none", "ultrahigh", "HIGH", ""))
def test_prepare_rejects_unsupported_codex_reasoning_effort_before_writing_prompt(
    tmp_path: Path, effort: str
) -> None:
    prompt_path = (tmp_path / "invalid-reasoning-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"reasoning_effort": effort}
                    )
                }
            )
        }
    )

    with pytest.raises(SetupError, match="reasoning effort"):
        RSILoopAgentAdapter(RSILoopConfig()).prepare(
            AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path)
        )

    assert not prompt_path.exists()


def test_install_hooks_uses_rsi_loop_hook_through_engine_runtime(
    tmp_path: Path,
) -> None:
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    container = ContainerRef(container_id="work-1", role="work")
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)

    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=container,
            submit_url="http://control.internal",
            token="runtime-only-token",
        )
    )

    targets = tuple(target for _, _, target in runtime.copies)
    assert PurePosixPath("/tmp/rsi-loop-codex-stop-hook.sh") in targets
    assert PurePosixPath("/etc/codex/hooks.json") in targets
    assert all(
        "runtime-only-token" not in source.read_text()
        for _, source, _ in runtime.copies
    )


@pytest.mark.parametrize(
    "agent_name,settings_path",
    [
        ("claude-code", "/home/agent/.claude/settings.json"),
        ("codex", "/etc/codex/hooks.json"),
    ],
)
@pytest.mark.parametrize("read_only_copy", [True, False])
def test_stop_hook_runs_after_read_only_or_mode_reset_copy(
    tmp_path: Path,
    read_only_copy: bool,
    agent_name: str,
    settings_path: str,
) -> None:
    """Apptainer binds the source read-only; Docker installs a 0644 copy."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    chmod_attempt = tmp_path / "chmod-attempt"
    if read_only_copy:
        chmod = binaries / "chmod"
        chmod.write_text(
            "#!/bin/sh\n"
            f"touch {shlex.quote(str(chmod_attempt))}\n"
            "echo 'Read-only file system' >&2\nexit 1\n"
        )
        chmod.chmod(0o755)

    class LocalRuntime(RecordingAgentRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.targets: dict[PurePosixPath, Path] = {}

        def copy_to(self, container, source, target):
            super().copy_to(container, source, target)
            destination = tmp_path / "container" / str(target).lstrip("/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if not read_only_copy:
                destination.chmod(0o644)
            self.targets[target] = destination

        def exec(self, container, command, **kwargs):
            super().exec(container, command, **kwargs)
            if not any(
                str(target) in str(command) and str(target).endswith("-stop-hook.sh")
                for target in self.targets
            ):
                # The existing Codex configuration setup is outside this test.
                return AgentRunResult(exit_code=0)
            argv = (
                ["/bin/sh", "-c", command]
                if isinstance(command, str)
                else list(command)
            )
            for target, destination in self.targets.items():
                argv = [part.replace(str(target), str(destination)) for part in argv]
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
            )
            return AgentRunResult(
                exit_code=result.returncode,
                output=result.stdout + result.stderr,
            )

    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(update={"name": agent_name}),
                }
            )
        }
    )
    runtime = LocalRuntime()
    previous_umask = os.umask(0o077)
    try:
        RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime).install_hooks(
            AgentHookRequest(
                run_plan=plan,
                container=ContainerRef(container_id="work-1", role="work"),
                submit_url="http://control.internal",
                token="runtime-only-token",
            )
        )
    finally:
        os.umask(previous_umask)

    assert runtime.executions[1]["user"] is None
    settings = json.loads(runtime.targets[PurePosixPath(settings_path)].read_text())
    command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    hook = runtime.targets[PurePosixPath(command)]
    completed = subprocess.run(
        [str(hook)],
        input="{}",
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "decision": "block",
        "reason": "Do not stop. Continue working on the implementation.",
    }
    assert hook.stat().st_mode & 0o777 == 0o755
    assert not chmod_attempt.exists()


@pytest.mark.parametrize(
    "agent_name,settings_filename",
    [
        ("claude-code", "_claude_settings.json"),
        ("codex", "_codex_hooks.json"),
    ],
)
@pytest.mark.parametrize("successful_checks", [0, 1])
def test_stop_hook_setup_failure_does_not_register_hook(
    tmp_path: Path,
    successful_checks: int,
    agent_name: str,
    settings_filename: str,
) -> None:
    class FailingRuntime(RecordingAgentRuntime):
        def exec(self, container, command, **kwargs):
            super().exec(container, command, **kwargs)
            return AgentRunResult(
                exit_code=0 if len(self.executions) <= successful_checks else 1,
                output="hook permissions unavailable",
            )

    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(update={"name": agent_name}),
                }
            )
        }
    )
    runtime = FailingRuntime()
    with pytest.raises(RuntimeError):
        RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime).install_hooks(
            AgentHookRequest(
                run_plan=plan,
                container=ContainerRef(container_id="work-1", role="work"),
                submit_url="http://control.internal",
                token="runtime-only-token",
            )
        )
    assert not (plan.paths.logs / settings_filename).exists()


def test_disabled_stop_hook_keeps_control_binding_without_container_mutation(
    tmp_path: Path,
) -> None:
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"install_stop_hook": False}
                    )
                }
            )
        }
    )
    container = ContainerRef(container_id="work-1", role="work")
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)

    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=container,
            submit_url="http://control.internal:8123",
            token="runtime-only-token",
        )
    )

    assert runtime.copies == []
    assert runtime.executions == []
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )
    adapter.run(AgentRunRequest(prepared=prepared, container=container))
    assert runtime.executions[-1]["environment"] == {
        **dict(prepared.environment),
        "RSI_JUDGE_URL": "http://control.internal:8123",
        "RSI_TOKEN": "runtime-only-token",
    }


def test_run_copies_prompt_and_executes_prepared_command(tmp_path: Path) -> None:
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=ContainerRef(container_id="work-1", role="work"),
            submit_url="http://control.internal:8123",
            token="runtime-only-token",
        )
    )
    runtime.result = AgentRunResult(exit_code=7, output="agent output")
    request = AgentRunRequest(
        prepared=prepared,
        container=ContainerRef(container_id="work-1", role="work"),
        timeout_seconds=12.5,
    )

    result = adapter.run(request)

    assert runtime.copies[-1][2] == PurePosixPath("/tmp/rsi-agent-prompt.md")
    assert runtime.executions[-1] == {
        "container": "work-1",
        "command": prepared.command,
        "timeout_seconds": 12.5,
        "user": None,
        "environment": {
            **dict(prepared.environment),
            "RSI_JUDGE_URL": "http://control.internal:8123",
            "RSI_TOKEN": "runtime-only-token",
        },
    }
    assert result == AgentRunResult(exit_code=7, output="agent output")


def test_run_forwards_exec_start_callback_only_after_prompt_copy(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class DeadlineAwareRuntime(RecordingAgentRuntime):
        def copy_to(
            self,
            container: ContainerRef,
            source: Path,
            target: PurePosixPath,
        ) -> None:
            super().copy_to(container, source, target)
            events.append("prompt_copied")

        def exec(self, *args: object, on_exec_start=None, **kwargs: object):
            events.append("exec_entered")
            assert on_exec_start is not None
            on_exec_start(4321.5)
            return self.result

    runtime = DeadlineAwareRuntime()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"install_stop_hook": False}
                    )
                }
            )
        }
    )
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )
    container = ContainerRef(container_id="work-1", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=container,
            submit_url="http://control.internal:8123",
            token="runtime-only-token",
        )
    )

    def record_deadline(deadline: float) -> None:
        events.append(f"deadline:{deadline}")

    adapter.run(
        AgentRunRequest(
            prepared=prepared,
            container=container,
            timeout_seconds=12.5,
            on_exec_start=record_deadline,
        )
    )

    assert events == ["prompt_copied", "exec_entered", "deadline:4321.5"]


def test_run_live_output_callback_is_protected_by_runtime_redaction(
    tmp_path: Path,
) -> None:
    runtime = RecordingAgentRuntime()
    config = RSILoopConfig(agent_api_key="provider-secret")
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(config, runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )
    container = ContainerRef(container_id="work-1", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=container,
            submit_url="http://control.internal:8123",
            token="runtime-only-token",
        )
    )

    def callback(_: str) -> None:
        pass

    adapter.run(
        AgentRunRequest(
            prepared=prepared,
            container=container,
            output_callback=callback,
        )
    )

    execution = runtime.executions[-1]
    assert execution["output_callback"] is callback
    assert set(execution["output_redact_values"]) >= {
        "provider-secret",
        "runtime-only-token",
        "http://control.internal:8123",
    }


def test_run_rejects_missing_control_hook_before_container_mutation(
    tmp_path: Path,
) -> None:
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )

    with pytest.raises(InfrastructureError, match="control environment"):
        adapter.run(
            AgentRunRequest(
                prepared=prepared,
                container=ContainerRef(container_id="work-1", role="work"),
            )
        )

    assert runtime.copies == []
    assert runtime.executions == []


def test_control_environment_is_bound_to_exact_work_and_mismatch_is_fail_closed(
    tmp_path: Path,
) -> None:
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=ContainerRef(container_id="work-1", role="work"),
            submit_url="http://control.internal:8123",
            token="runtime-only-token",
        )
    )
    copies_before = len(runtime.copies)
    executions_before = len(runtime.executions)

    with pytest.raises(InfrastructureError, match="different Work container"):
        adapter.run(
            AgentRunRequest(
                prepared=prepared,
                container=ContainerRef(container_id="work-2", role="work"),
            )
        )

    assert len(runtime.copies) == copies_before
    assert len(runtime.executions) == executions_before
    with pytest.raises(InfrastructureError, match="unavailable"):
        adapter.run(
            AgentRunRequest(
                prepared=prepared,
                container=ContainerRef(container_id="work-1", role="work"),
            )
        )


def test_control_environment_is_runtime_only_merged_redacted_and_cleared(
    tmp_path: Path,
) -> None:
    token = "runtime-only-control-secret"
    submit_url = "http://control.internal:8123"
    runtime = RecordingAgentRuntime()
    runtime.result = AgentRunResult(
        exit_code=0,
        output=(
            f"unrelated prefix {submit_url} {token} repeated {submit_url} "
            f"combined={submit_url}{token} unrelated suffix"
        ),
    )
    plan = make_run_plan(tmp_path)
    config = RSILoopConfig(
        agent_api_key="ordinary-agent-secret",
        agent_extra_env={
            "EXTRA": "ordinary-value",
            "RSI_JUDGE_URL": "http://attacker.invalid",
            "RSI_TOKEN": "attacker-token",
        },
    )
    adapter = RSILoopAgentAdapter(config, runtime=runtime)
    prompt_path = (tmp_path / "prompt.md").resolve()
    prepared = adapter.prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=prompt_path)
    )

    assert "RSI_JUDGE_URL" not in dict(prepared.environment)
    assert "RSI_TOKEN" not in dict(prepared.environment)
    assert token not in prompt_path.read_text()
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=ContainerRef(container_id="work-1", role="work"),
            submit_url=submit_url,
            token=token,
        )
    )
    assert all(token not in source.read_text() for _, source, _ in runtime.copies)

    result = adapter.run(
        AgentRunRequest(
            prepared=prepared,
            container=ContainerRef(container_id="work-1", role="work"),
            timeout_seconds=12.5,
        )
    )

    environment = runtime.executions[-1]["environment"]
    assert isinstance(environment, dict)
    assert environment["EXTRA"] == "ordinary-value"
    assert environment["CODEX_API_KEY"] == "ordinary-agent-secret"
    assert environment["RSI_JUDGE_URL"] == submit_url
    assert environment["RSI_TOKEN"] == token
    assert submit_url not in result.output
    assert token not in result.output
    assert result.output == (
        "unrelated prefix [REDACTED] [REDACTED] repeated [REDACTED] "
        "combined=[REDACTED][REDACTED] unrelated suffix"
    )
    assert "RSI_JUDGE_URL" not in dict(prepared.environment)
    assert "RSI_TOKEN" not in dict(prepared.environment)
    assert all(token not in source.read_text() for _, source, _ in runtime.copies)

    copies_before = len(runtime.copies)
    executions_before = len(runtime.executions)
    with pytest.raises(InfrastructureError, match="control environment"):
        adapter.run(
            AgentRunRequest(
                prepared=prepared,
                container=ContainerRef(container_id="work-1", role="work"),
            )
        )
    assert len(runtime.copies) == copies_before
    assert len(runtime.executions) == executions_before


def test_explicit_cleanup_clears_control_binding_before_agent_run(tmp_path) -> None:
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan, prompt_path=(tmp_path / "prompt.md").resolve()
        )
    )
    work = ContainerRef(container_id="work-1", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=work,
            submit_url="http://control.internal:8123",
            token="runtime-only-control-secret",
        )
    )
    executions_before = len(runtime.executions)

    adapter.clear_transient_bindings()

    with pytest.raises(InfrastructureError, match="unavailable"):
        adapter.run(AgentRunRequest(prepared=prepared, container=work))
    assert len(runtime.executions) == executions_before


def test_agent_runtime_exception_exact_redacts_control_and_provider_secrets(
    tmp_path,
) -> None:
    submit_url = "http://control.internal:8123"
    token = "runtime-only-control-secret"
    provider_secret = "runtime-only-provider-secret"
    runtime = RecordingAgentRuntime()
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(
        RSILoopConfig(agent_api_key=provider_secret), runtime=runtime
    )
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan, prompt_path=(tmp_path / "prompt.md").resolve()
        )
    )
    work = ContainerRef(container_id="work-1", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=work,
            submit_url=submit_url,
            token=token,
        )
    )

    def fail_exec(*_args, **_kwargs):
        raise RuntimeError(f"runtime echoed {submit_url} {token} {provider_secret}")

    runtime.exec = fail_exec  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError) as caught:
        adapter.run(AgentRunRequest(prepared=prepared, container=work))

    message = str(caught.value)
    assert submit_url not in message
    assert token not in message
    assert provider_secret not in message
    assert "[REDACTED]" in message
    with pytest.raises(InfrastructureError, match="unavailable"):
        adapter.run(AgentRunRequest(prepared=prepared, container=work))


def test_run_redacts_bare_provider_proxy_and_custom_secret_values_longest_first(
    tmp_path: Path,
) -> None:
    provider_secret = "opaque-provider-value"
    proxy_secret = "http://proxy-user:proxy-password@proxy.internal:8080"
    custom_secret = "opaque-custom-value"
    runtime = RecordingAgentRuntime()
    runtime.result = AgentRunResult(
        exit_code=0,
        output=(
            f"{provider_secret} {proxy_secret} proxy-user proxy-password "
            f"{custom_secret} "
            f"nested={custom_secret}{provider_secret} harmless-value"
        ),
    )
    plan = make_run_plan(tmp_path)
    adapter = RSILoopAgentAdapter(
        RSILoopConfig(
            agent_api_key=provider_secret,
            https_proxy=proxy_secret,
            agent_extra_env={"CUSTOM_SECRET": custom_secret},
        ),
        runtime=runtime,
    )
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )
    work = ContainerRef(container_id="work-1", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=work,
            submit_url="http://control.internal:8123",
            token="control-secret",
        )
    )

    result = adapter.run(AgentRunRequest(prepared=prepared, container=work))

    assert result.output == (
        "[REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] "
        "nested=[REDACTED][REDACTED] harmless-value"
    )


def test_codex_names_the_api_base_url_provider_at_launch() -> None:
    from rsi_loop.harness.agent import create_agent

    proxied = create_agent(
        "codex", RSILoopConfig(agent_api_base_url="http://model.test:8000/v1")
    ).format_run_cmd("/tmp/prompt.md", model="m")
    direct = create_agent("codex", RSILoopConfig()).format_run_cmd(
        "/tmp/prompt.md", model="m"
    )

    argv = shlex.split(proxied.split('"$(cat')[0])
    assert "model_provider=rsi_proxy" in argv
    assert 'model_providers.rsi_proxy.base_url="$OPENAI_BASE_URL"' in argv
    assert "model_providers.rsi_proxy.env_key=OPENAI_API_KEY" in argv
    # The URL and key stay in the environment, not in the command line.
    assert "model.test" not in proxied
    assert "model_provider" not in direct


# -- one-shot commands for rsi-agent in a Judge ------------------------------------


def _oneshot(
    agent, *, base_url=None, internet=False, effort="high", model="m", **config
):
    from rsi_harness.integrations.rsi_loop import rsi_loop_oneshot
    from rsi_harness.models import AgentPlan, NetworkPolicy

    return rsi_loop_oneshot(
        RSILoopConfig(
            agent_api_key="sk-oneshot", agent_api_base_url=base_url, **config
        ),
        AgentPlan(
            name=agent,
            model=model,
            reasoning_effort=effort,
            network=NetworkPolicy(mode="public" if internet else "no-network"),
        ),
    )


CODEX_ONESHOT = (
    "--ephemeral --skip-git-repo-check --ignore-user-config --ignore-rules"
    " --dangerously-bypass-approvals-and-sandbox -c features.hooks=false"
    " -c features.goals=false -c project_doc_max_bytes=0"
    " --output-last-message {output_file} {schema_option} -"
)


@pytest.mark.parametrize(("model", "effort"), [("m", "high"), ("gpt-6-astra", "max")])
def test_codex_one_shot_command_matches_work_launch_flags(
    model: str, effort: str
) -> None:
    offline = _oneshot("codex", model=model, effort=effort)
    assert offline["command"] == (
        f"codex exec -c 'model_reasoning_effort=\"{effort}\"' "
        '-c web_search="disabled" -c allow_login_shell=false '
        f"-c features.shell_snapshot=false --model {model} " + CODEX_ONESHOT
    )
    assert offline["model"] == model
    assert offline["reasoning_effort"] == effort
    assert offline["schema_option"] == "--output-schema {schema_file}"
    assert offline["output"] == {"mode": "file", "fields": [], "error_field": None}
    assert offline["unsupported"] is None
    assert offline["environment"]["CODEX_API_KEY"] == "sk-oneshot"

    online = _oneshot("codex", internet=True, effort=None, model=None)
    assert online["command"] == (
        "codex exec -c allow_login_shell=false -c features.shell_snapshot=false "
        + CODEX_ONESHOT
    )


def test_codex_one_shot_names_the_base_url_provider_at_launch() -> None:
    proxied = _oneshot("codex", base_url="http://model.test:8000/v1", internet=True)

    command = proxied["command"]
    argv = shlex.split(command.replace("{", "<").replace("}", ">"))
    assert "model_provider=rsi_proxy" in argv
    assert 'model_providers.rsi_proxy.base_url="$OPENAI_BASE_URL"' in argv
    assert "model_providers.rsi_proxy.env_key=OPENAI_API_KEY" in argv
    assert "model.test" not in command and "sk-oneshot" not in command
    assert proxied["environment"]["OPENAI_BASE_URL"] == "http://model.test:8000/v1"
    assert "CODEX_API_KEY" not in proxied["environment"]


def test_claude_one_shot_command_prints_one_json_result() -> None:
    offline = _oneshot("claude-code", effort="max")
    assert offline["command"] == (
        "claude --effort max -p --output-format json --no-session-persistence"
        " --dangerously-skip-permissions --safe-mode --setting-sources user"
        " {schema_option} --model m --disallowedTools WebSearch,WebFetch"
    )
    assert offline["schema_option"] == '--json-schema "$(cat {schema_file})"'
    assert offline["output"] == {
        "mode": "stdout-json",
        "fields": ["structured_output", "result"],
        "error_field": "is_error",
    }
    assert offline["environment"]["ANTHROPIC_AUTH_TOKEN"] == "sk-oneshot"
    assert offline["environment"]["IS_SANDBOX"] == "1"

    online = _oneshot("claude-code", internet=True, effort=None)
    assert "--disallowedTools" not in online["command"]
    assert online["command"].startswith("claude -p ")


MODEL_LIMITS = {"agent_context_window": "131072", "agent_max_output_tokens": "16384"}
QWEN_CONFIRMED = (
    "read_file,glob,grep_search,run_shell_command,edit,write_file,notebook_edit"
)
QWEN_DENIED = "enter_plan_mode,exit_plan_mode,ask_user_question"


def test_qwen_one_shot_command_prints_one_result_line() -> None:
    base_url = "http://model.test:8000/v1"
    offline = _oneshot("qwen-code", base_url=base_url, effort=None, **MODEL_LIMITS)

    prelude, command = offline["command"].split("\n")
    # Print mode, never the driver; the prompt arrives on stdin.
    assert command == (
        "qwen --safe-mode --chat-recording=false --output-format stream-json"
        f" --allowed-tools {QWEN_CONFIRMED}"
        # The skill tool too: it would list the working directory's
        # .qwen/commands.
        f" --exclude-tools {QWEN_DENIED},web_fetch,web_search,skill"
        " --model m {schema_option} | tail -n 1"
    )
    # Work's settings, with folder trust on, written into the fresh HOME.
    agent = create_agent("qwen-code", RSILoopConfig(**MODEL_LIMITS))
    settings = agent.oneshot_settings(internet=False)
    assert settings == {
        **agent.settings(internet=False),
        "security": {
            "auth": {"selectedType": "openai"},
            "folderTrust": {"enabled": True},
        },
    }
    assert prelude == (
        "mkdir -p \"$HOME/.qwen\" && printf '%s\\n' "
        + shlex.quote(json.dumps(settings, sort_keys=True))
        + ' > "$HOME/.qwen/settings.json" && set -o pipefail &&'
    )
    assert offline["schema_option"] == "--json-schema @{schema_file}"
    assert offline["output"] == {
        "mode": "stdout-json",
        "fields": ["structured_result", "result"],
        "error_field": "is_error",
    }
    environment = offline["environment"]
    assert environment["OPENAI_API_KEY"] == "sk-oneshot"
    assert environment["OPENAI_BASE_URL"] == base_url
    assert environment["OPENAI_MODEL"] == "m"
    assert environment["QWEN_CODE_MODELS_DEV"] == "off"
    assert environment["QWEN_TELEMETRY_ENABLED"] == "false"
    assert "sk-oneshot" not in offline["command"]
    assert "model.test" not in offline["command"]

    online = _oneshot("qwen-code", internet=True, effort=None, **MODEL_LIMITS)
    argv = shlex.split(online["command"].split("\n")[1])
    assert argv[argv.index("--allowed-tools") + 1] == (
        f"{QWEN_CONFIRMED},web_fetch,web_search"
    )
    assert argv[argv.index("--exclude-tools") + 1] == f"{QWEN_DENIED},skill"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        # As in Work: Qwen takes no reasoning effort yet.
        ({**MODEL_LIMITS, "effort": "high"}, "does not support reasoning effort"),
        ({**MODEL_LIMITS, "effort": None, "model": None}, "needs a model"),
        (
            {"effort": None, "agent_context_window": "131072"},
            "RSI_AGENT_MAX_OUTPUT_TOKENS is not set",
        ),
    ],
)
def test_qwen_one_shot_refuses_what_work_refuses(options, message) -> None:
    with pytest.raises(SetupError, match=message):
        _oneshot("qwen-code", **options)


def test_kimi_one_shot_command_uses_print_mode_with_work_config() -> None:
    base_url = "http://model.test:8000/v1"
    offline = _oneshot("kimi-code", base_url=base_url, **MODEL_LIMITS)

    command = offline["command"]
    lines = command.split("\n")
    # Work's config.toml in the fresh HOME, with the reasoning effort.
    agent = create_agent("kimi-code", RSILoopConfig(**MODEL_LIMITS))
    prelude = agent.config_prelude(model="m", internet=False)
    assert command.startswith(agent.apply_reasoning_effort(prelude, "high") + " &&\n")
    assert 'default_effort = "high"' in command
    assert '"FetchURL", "WebSearch"' in command
    assert lines[-3:] == [
        "unset KIMI_CODE_INFINITE_RETRY && set -o pipefail &&",
        "{ kimi=$(command -v kimi) || exit 127; } &&",
        lines[-1],
    ]
    # Print mode, never the driver; the prompt arrives on stdin, and only
    # the last assistant line leaves the pipe.
    launch, keep = lines[-1].split(" | ")
    argv = shlex.split(launch)
    assert argv[:2] == [NODE_BIN, "-e"]
    assert "readFileSync(0" in argv[2] and '"--prompt=" + prompt' in argv[2]
    assert argv[3:] == [
        "--",
        "$kimi",
        "--output-format",
        "stream-json",
        "--agent-file",
        "$KIMI_CODE_HOME/oneshot-agent.md",
        "--skills-dir",
        "$KIMI_CODE_HOME/skills",
        "{schema_option}",
    ]
    assert keep == 'sed -n \'/^{"role":"assistant"/h;${x;p;}\''
    # The explicit profile: Kimi's default system prompt, no subagents.
    assert (
        "printf '%s' '---\nname: agent\ndescription: Default agent\n"
        "disallowedTools: Agent, AgentSwarm\n---\n${base_prompt}\n'"
        ' > "$KIMI_CODE_HOME/oneshot-agent.md" &&'
    ) in command
    # No native schema option: rsi-agent states the schema in the prompt.
    assert offline["schema_option"] is None
    assert offline["output"] == {
        "mode": "stdout-json",
        "fields": ["content"],
        "error_field": None,
    }
    environment = offline["environment"]
    assert environment["OPENAI_API_KEY"] == "sk-oneshot"
    assert environment["OPENAI_BASE_URL"] == base_url
    assert environment["KIMI_DISABLE_TELEMETRY"] == "1"
    assert environment["KIMI_CODE_NO_AUTO_UPDATE"] == "1"
    assert "sk-oneshot" not in command
    assert "model.test" not in command

    online = _oneshot("kimi-code", internet=True, effort=None, **MODEL_LIMITS)
    assert '"FetchURL"' not in online["command"]
    assert 'capabilities = ["tool_use"]\n' in online["command"]
    assert "default_effort" not in online["command"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        # As in Work: an effort Kimi does not declare.
        ({**MODEL_LIMITS, "effort": "xhigh"}, "unsupported Kimi Code reasoning effort"),
        ({**MODEL_LIMITS, "model": None}, "needs a model"),
        ({"agent_context_window": "131072"}, "RSI_AGENT_MAX_OUTPUT_TOKENS is not set"),
    ],
)
def test_kimi_one_shot_refuses_what_work_refuses(options, message) -> None:
    with pytest.raises(SetupError, match=message):
        _oneshot("kimi-code", **options)


def test_opencode_one_shot_command_uses_run_with_work_config() -> None:
    base_url = "http://model.test:8000/v1"
    offline = _oneshot("opencode", base_url=base_url, **MODEL_LIMITS)

    command = offline["command"]
    lines = command.split("\n")
    # Work's config in the fresh HOME, with the reasoning effort.
    agent = create_agent("opencode", RSILoopConfig(**MODEL_LIMITS))
    prelude = agent.config_prelude(model="m", internet=False)
    assert command.startswith(agent.apply_reasoning_effort(prelude, "high") + " &&\n")
    config = json.loads(shlex.split(lines[0].rsplit(" && ", 1)[1])[1].split("=", 1)[1])
    rsi_model = config["provider"]["rsi"]["models"]["m"]
    assert rsi_model["options"] == {"reasoningEffort": "high"}
    assert rsi_model["limit"] == {"context": 131072, "output": 16384}
    assert config["permission"]["webfetch"] == "deny"
    # Temporary files in the private HOME, not the Judge's /tmp.
    assert lines[1] == 'export TMPDIR="$HOME/.tmp" && mkdir -p "$TMPDIR" &&'
    # Nothing from the working directory: project config, skills, CLAUDE.md.
    assert lines[2] == (
        "export OPENCODE_DISABLE_PROJECT_CONFIG=1"
        " OPENCODE_DISABLE_EXTERNAL_SKILLS=1 OPENCODE_DISABLE_CLAUDE_CODE=1"
        " && set -o pipefail &&"
    )
    # Print mode, never the driver; the prompt arrives on stdin, errors go
    # to stderr and only the last text part leaves the pipe.
    launch, keep = lines[3].split(" | ")
    assert launch == "opencode run --format json {schema_option}"
    assert shlex.split(keep) == [
        "sed",
        "-n",
        "-e",
        '/^{"type":"error",/w /dev/stderr',
        "-e",
        '/^{"type":"text",/h;${x;s/^{"type":"text","timestamp":[0-9]*,'
        r'"sessionID":"[^"]*","part":\(.*\)}$/\1/p;}',
    ]
    assert len(lines) == 4
    # No structured-output option: rsi-agent states the schema in the prompt.
    assert offline["schema_option"] is None
    assert offline["output"] == {
        "mode": "stdout-json",
        "fields": ["text"],
        "error_field": None,
    }
    # The config names the key and endpoint variables, never their values.
    environment = offline["environment"]
    assert environment["RSI_AGENT_API_KEY"] == "sk-oneshot"
    assert environment["RSI_AGENT_API_BASE_URL"] == base_url
    assert environment["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    assert environment["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert "sk-oneshot" not in command
    assert "model.test" not in command

    online = _oneshot("opencode", internet=True, effort=None, **MODEL_LIMITS)
    assert '"webfetch": "deny"' not in online["command"]
    assert '"options": {}' in online["command"]
    assert "reasoningEffort" not in online["command"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        # As in Work: an effort OpenCode does not declare.
        ({**MODEL_LIMITS, "effort": "xhigh"}, "unsupported OpenCode reasoning effort"),
        ({**MODEL_LIMITS, "model": None}, "needs a model"),
        ({"agent_context_window": "131072"}, "RSI_AGENT_MAX_OUTPUT_TOKENS is not set"),
    ],
)
def test_opencode_one_shot_refuses_what_work_refuses(options, message) -> None:
    with pytest.raises(SetupError, match=message):
        _oneshot("opencode", **options)


ZCODE_ONESHOT_DENIED = (
    "AskUserQuestion,EnterPlanMode,ExitPlanMode,CronList,CronUpdate,"
    "CronCreate,CronDelete,OffPeakCreate,Skill,Agent,SendMessage"
)


def test_zcode_one_shot_command_uses_print_mode_with_work_config() -> None:
    base_url = "http://model.test:8000/v1"
    offline = _oneshot("zcode", base_url=base_url, **MODEL_LIMITS)

    command = offline["command"]
    # Work's CLI and provider config, with the reasoning effort; the provider
    # config, which will hold the key, goes into the fresh HOME, which must
    # be a tmpfs, instead of a directory under /dev/shm.
    agent = create_agent("zcode", RSILoopConfig(**MODEL_LIMITS))
    prelude = agent.apply_reasoning_effort(
        agent.config_prelude(model="m", internet=False), "high"
    )
    work_secret = (
        'test "$(stat -f -c %T /dev/shm)" = tmpfs'
        " && rsi_secret_dir=$(mktemp -d /dev/shm/rsi-zcode.XXXXXX)"
    )
    assert work_secret in prelude
    prelude = prelude.replace(
        work_secret,
        'test "$(stat -f -c %T "$HOME")" = tmpfs && rsi_secret_dir="$HOME/.zcode"',
    )
    assert command.startswith(prelude + " &&\n")
    assert '"values": ["high"]' in prelude
    assert '"options": {"reasoningLevel": "high"}' in prelude
    lines = command.removeprefix(prelude + " &&\n").split("\n")
    # The working directory's config and .env cannot pick a proxy or CAs,
    # unbound the retries or move ZCode's files out of the private HOME.
    kept = (
        "ZCODE_HTTP_PROXY",
        "ZCODE_NO_PROXY",
        "ZCODE_AGENT_CA_CERT",
        "NODE_TLS_REJECT_UNAUTHORIZED",
        "ZCODE_MODEL_RETRY_MAX_RETRIES",
        "ZCODE_MODEL_RETRY_BASE_DELAY_MS",
        "ZCODE_MODEL_RETRY_MAX_DELAY_MS",
        "ZCODE_MODEL_RETRY_BACKOFF_FACTOR",
        "ZCODE_BETA",
        "ZCODE_ENV",
    )
    assert lines[0] == (
        "export "
        + "".join(f'{name}="${{{name}-}}" ' for name in kept)
        + 'ZCODE_STORAGE_DIR="$HOME/.zcode" ZCODE_HOME="$HOME/.zcode"'
        ' ZCODE_DATA_BASE_DIR="$HOME" ZCODE_LOG_DIR="$HOME/.zcode/cli/log" &&'
    )
    assert lines[1] == "{ test -f /opt/zcode/zcode.cjs || exit 127; } &&"
    # Print mode under the pinned node, never the driver; the prompt arrives
    # on stdin and reaches ZCode's argv only inside its own process.
    argv = shlex.split(lines[2])
    assert argv[:2] == ["/usr/local/bin/node", "-e"]
    assert "readFileSync(0" in argv[2] and '"--prompt=" + prompt' in argv[2]
    assert argv[3:] == [
        "--",
        "/opt/zcode/zcode.cjs",
        "--output-format",
        "json",
        "--mode",
        "yolo",
        "--disallowed-tools",
        ZCODE_ONESHOT_DENIED + ",WebFetch",
        "{schema_option}",
    ]
    assert len(lines) == 3
    # No structured-output option: rsi-agent states the schema in the prompt.
    assert offline["schema_option"] is None
    assert offline["output"] == {
        "mode": "stdout-json",
        "fields": ["response"],
        "error_field": None,
    }
    # The prelude copies the key and endpoint from the environment.
    environment = offline["environment"]
    assert environment["RSI_AGENT_API_KEY"] == "sk-oneshot"
    assert environment["RSI_AGENT_API_BASE_URL"] == base_url
    assert environment["ZCODE_BASE_URL"] == "http://127.0.0.1:9"
    assert environment["ZCODE_MODEL_TELEMETRY_ENABLED"] == "false"
    assert "sk-oneshot" not in command
    assert "model.test" not in command

    online = _oneshot("zcode", internet=True, effort=None, **MODEL_LIMITS)
    assert shlex.split(online["command"].split("\n")[-1])[-2] == ZCODE_ONESHOT_DENIED
    assert '"values": ["default"], "map": "{}"' in online["command"]
    assert '"options": {"reasoningLevel": "default"}' in online["command"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        # As in Work: an effort ZCode does not declare.
        ({**MODEL_LIMITS, "effort": "xhigh"}, "unsupported ZCode reasoning effort"),
        ({**MODEL_LIMITS, "model": None}, "needs a model"),
        ({"agent_context_window": "131072"}, "RSI_AGENT_MAX_OUTPUT_TOKENS is not set"),
    ],
)
def test_zcode_one_shot_refuses_what_work_refuses(options, message) -> None:
    with pytest.raises(SetupError, match=message):
        _oneshot("zcode", **options)


PI_ONESHOT_FLAGS = [
    "--no-session",
    "--no-approve",
    "--no-context-files",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
]


def test_pi_one_shot_command_uses_print_mode_with_work_config() -> None:
    base_url = "http://model.test:8000/v1"
    offline = _oneshot("pi", base_url=base_url, **MODEL_LIMITS)

    command = offline["command"]
    lines = command.split("\n")
    # Work's settings and models in the fresh HOME, with the reasoning
    # effort turned on: Work's own command up to its launch.
    agent = create_agent("pi", RSILoopConfig(**MODEL_LIMITS))
    work = agent.apply_reasoning_effort(
        agent.format_run_cmd("/p", model="m", continuation="go on"), "high"
    )
    prelude = work.split(" && export RSI_CONTINUATION_REASON=")[0]
    assert command.startswith(prelude + " &&\n")
    assert command.removeprefix(prelude).count("\n") == 3
    models = json.loads(shlex.split(prelude.rsplit(" && ", 1)[1])[3])
    rsi_model = models["providers"]["rsi"]["models"][0]
    assert rsi_model["reasoning"] is True
    assert (rsi_model["contextWindow"], rsi_model["maxTokens"]) == (131072, 16384)
    assert models["providers"]["rsi"]["apiKey"] == "$RSI_AGENT_API_KEY"
    # Temporary files in the private HOME, not the Judge's /tmp.
    assert lines[-3] == 'export TMPDIR="$HOME/.tmp" && mkdir -p "$TMPDIR" &&'
    # Print mode without the extension that keeps Work going, and nothing
    # from the working directory; the prompt arrives on stdin.
    assert lines[-2] == (
        "answer=$(exec "
        + shlex.join(
            ["pi", "--thinking", "high", "--offline", "-p", *PI_ONESHOT_FLAGS]
            + ["--provider", "rsi", "--model", "m"]
        )
        + " {schema_option}) &&"
    )
    assert "rsi-never-stop" not in command
    # The final message without print mode's trailing newline.
    assert lines[-1] == 'printf "%s" "$answer" >{output_file}'
    # No structured-output option: rsi-agent states the schema in the prompt.
    assert offline["schema_option"] is None
    assert offline["output"] == {"mode": "file", "fields": [], "error_field": None}
    # The config names the key and endpoint variables, never their values.
    environment = offline["environment"]
    assert environment["RSI_AGENT_API_KEY"] == "sk-oneshot"
    assert environment["RSI_AGENT_API_BASE_URL"] == base_url
    assert environment["PI_OFFLINE"] == "1"
    assert environment["PI_SKIP_VERSION_CHECK"] == "1"
    assert environment["PI_TELEMETRY"] == "0"
    assert "sk-oneshot" not in command
    assert "model.test" not in command

    # pi has no web tools: online differs only in the environment.
    online = _oneshot("pi", internet=True, effort=None, **MODEL_LIMITS)
    assert online["command"] == agent.format_oneshot_cmd(model="m").command
    assert '"reasoning": false' in online["command"]
    assert "--thinking" not in online["command"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        # As in Work: an effort pi does not declare.
        ({**MODEL_LIMITS, "effort": "xhigh"}, "unsupported pi reasoning effort"),
        ({**MODEL_LIMITS, "model": None}, "needs a model"),
        ({"agent_context_window": "131072"}, "RSI_AGENT_MAX_OUTPUT_TOKENS is not set"),
    ],
)
def test_pi_one_shot_refuses_what_work_refuses(options, message) -> None:
    with pytest.raises(SetupError, match=message):
        _oneshot("pi", **options)


def test_an_agent_without_a_one_shot_command_is_named(monkeypatch) -> None:
    from rsi_loop.harness.agent.claude_code import ClaudeCodeAgent

    monkeypatch.setattr(
        ClaudeCodeAgent, "format_oneshot_cmd", lambda self, **options: None
    )

    result = _oneshot("claude-code")

    assert result["command"] is None
    assert result["unsupported"] == (
        "the run's Agent 'claude-code' cannot run one-shot in a Judge"
    )


def test_every_registered_agent_has_a_one_shot_command() -> None:
    from rsi_loop.harness.agent import OneShotCommand, create_agent

    # A configuration every Agent's setup accepts.
    config = RSILoopConfig(
        agent_api_key="sk-oneshot",
        agent_api_base_url="http://model.test:8000/v1",
        agent_model="m",
        **MODEL_LIMITS,
    )
    for agent_class in list_agent_classes():
        agent = create_agent(agent_class.name, config)
        agent.check_config()
        for internet in (False, True):
            oneshot = agent.format_oneshot_cmd(model="m", internet=internet)
            if agent_class.runs_task_solution:
                # The oracle has no model to call.
                assert oneshot is None, agent_class.name
            else:
                assert isinstance(oneshot, OneShotCommand), agent_class.name


def test_one_shot_commands_reject_an_incomplete_contract() -> None:
    from rsi_loop.harness.agent import OneShotCommand

    with pytest.raises(ValueError, match="output_file"):
        OneShotCommand(command="agent {schema_option}")
    with pytest.raises(ValueError, match="schema_option"):
        OneShotCommand(command="agent > {output_file}")
    with pytest.raises(ValueError, match="output fields"):
        OneShotCommand(command="agent {schema_option}", output="stdout-json")


def _driven_plan(tmp_path: Path, *, install_stop_hook: bool = True):
    plan = make_run_plan(tmp_path)
    return plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "name": "fake-driven",
                            "model": "m",
                            "install_stop_hook": install_stop_hook,
                        }
                    )
                }
            )
        }
    )


_LIMITS = {"agent_context_window": "128000", "agent_max_output_tokens": "8192"}


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("name", sorted(c.name for c in list_agent_classes()))
def test_every_agent_starts_outside_a_login_shell(
    tmp_path: Path, name: str, resume: bool
) -> None:
    """A login profile may reset PATH (Debian's and Alpine's do), dropping the
    image's ENV PATH and the sandbox client directory the exec prepends. No
    Agent start, resume or driver goes through one."""
    agent_class = next(c for c in list_agent_classes() if c.name == name)
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "name": name,
                            "model": None if agent_class.runs_task_solution else "m",
                        }
                    )
                }
            )
        }
    )
    config = RSILoopConfig(
        agent_api_key="sk-test",
        agent_api_base_url="http://203.0.113.7:8000/v1",
        **_LIMITS,
    )

    prepared = RSILoopAgentAdapter(config).prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "prompt.md").resolve(),
            resume=resume,
        )
    )

    assert AGENT_SHELL == ("/bin/bash", "-c")
    assert prepared.command[:2] == AGENT_SHELL
    assert len(prepared.command) == 3
    assert _BARE_NODE.search(prepared.command[2]) is None, prepared.command[2]


# `node` run by name: the first node on PATH, which may be a task image's.
_BARE_NODE = re.compile(r"(?:^|[\s;&|(])node\s")


@pytest.mark.parametrize("name", sorted(c.name for c in list_agent_classes()))
def test_every_agent_one_shot_runs_node_by_its_pinned_path(name: str) -> None:
    """A task image may put its own node ahead of /usr/local/bin."""
    config = RSILoopConfig(
        agent_api_key="sk-test",
        agent_api_base_url="http://203.0.113.7:8000/v1",
        **_LIMITS,
    )
    oneshot = create_agent(name, config).format_oneshot_cmd(model="m")
    if oneshot is None:
        pytest.skip(f"{name} has no one-shot command")

    assert _BARE_NODE.search(oneshot.command) is None, oneshot.command
    if name == "kimi-code":
        assert f"{NODE_BIN} -e" in oneshot.command


def test_codex_runs_the_models_commands_outside_a_login_shell() -> None:
    """Codex starts each command in a login shell unless told not to; on
    Debian that would drop the PATH its own process has."""
    agent = create_agent("codex", RSILoopConfig())
    oneshot = agent.format_oneshot_cmd(model="m")
    assert oneshot is not None
    commands = (
        agent.format_run_cmd("/tmp/prompt.md", model="m"),
        agent.format_run_cmd("/tmp/prompt.md", model="m", resume=True),
        oneshot.command,
    )

    for command in commands:
        assert command.startswith("codex exec ")
        assert command.count("-c allow_login_shell=false") == 1, command


@pytest.mark.parametrize("install_stop_hook", [True, False])
def test_prepare_passes_the_continuation_only_with_the_stop_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_stop_hook: bool
) -> None:
    register_fake_driven_agent(monkeypatch)

    prepared = RSILoopAgentAdapter(RSILoopConfig(**_LIMITS)).prepare(
        AgentPrepareRequest(
            run_plan=_driven_plan(tmp_path, install_stop_hook=install_stop_hook),
            prompt_path=(tmp_path / "prompt.md").resolve(),
        )
    )

    command = prepared.command[2]
    assert prepared.command[:2] == ("/bin/bash", "-c")
    assert "exec rsi-agent-driver --protocol acp" in command
    assert "--prompt-file /tmp/rsi-agent-prompt.md" in command
    expected = f"--continuation-reason {shlex.quote(STOP_HOOK_REASON)}"
    assert (expected in command) is install_stop_hook
    environment = dict(prepared.environment)
    assert environment["RSI_AGENT_CONTEXT_WINDOW"] == "128000"
    assert environment["RSI_AGENT_MAX_OUTPUT_TOKENS"] == "8192"


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"agent_context_window": "128000"},
        {"agent_context_window": "lots", "agent_max_output_tokens": "8192"},
    ],
)
def test_prepare_refuses_a_driven_agent_without_context_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limits: dict[str, str]
) -> None:
    register_fake_driven_agent(monkeypatch)
    prompt_path = (tmp_path / "prompt.md").resolve()

    with pytest.raises(SetupError, match="RSI_AGENT_CONTEXT_WINDOW"):
        RSILoopAgentAdapter(RSILoopConfig(**limits)).prepare(
            AgentPrepareRequest(
                run_plan=_driven_plan(tmp_path), prompt_path=prompt_path
            )
        )
    assert not prompt_path.exists()


@pytest.mark.parametrize("name", ["codex", "claude-code"])
def test_native_hook_agents_need_no_context_limits(name: str) -> None:
    validate_agent_config(RSILoopConfig(), name)


def test_unknown_agent_config_is_a_setup_error() -> None:
    with pytest.raises(SetupError, match="Unknown agent"):
        validate_agent_config(RSILoopConfig(), "no-such-agent")


@pytest.mark.parametrize(
    "name", sorted(c.name for c in list_agent_classes() if c.requires_model)
)
def test_production_refuses_a_run_without_a_model_before_any_runtime_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from rsi_harness.models import CompileOptions, RunRequest
    from rsi_harness.runtime import production
    from tests.factories import write_harbor_task

    class Reached(Exception):
        pass

    def reached(*_args: object, **_kwargs: object) -> None:
        raise Reached

    class Inventory:
        list_devices = reached

    services = production.ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=Inventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key="sk-test",
            agent_api_base_url="http://203.0.113.7:8000/v1",
            **_LIMITS,
        ),
    )
    monkeypatch.setattr(services, "_client", reached)
    monkeypatch.setattr(production, "resolve_agent_auth", reached)
    task = write_harbor_task(tmp_path)
    options = CompileOptions(agent_name=name)

    # A setup refusal, before any image, container or GPU work.
    with pytest.raises(SetupError, match="needs a model; set RSI_AGENT_MODEL"):
        services.run(RunRequest(task_dir=task, agent_name=name, options=options))
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()
    # The run's own model passes the check.
    with pytest.raises(Reached):
        services.run(
            RunRequest(task_dir=task, agent_name=name, model="m", options=options)
        )


def test_context_limits_load_from_rsi_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RSI_AGENT_CONTEXT_WINDOW", "200000")
    monkeypatch.setenv("RSI_AGENT_MAX_OUTPUT_TOKENS", "not-a-number")

    config = load_config()

    # Kept as given: only an Agent that needs them rejects a bad value.
    assert config.agent_context_window == "200000"
    assert config.agent_max_output_tokens == "not-a-number"
