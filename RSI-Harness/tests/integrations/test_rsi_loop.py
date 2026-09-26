from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
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
from rsi_loop.harness.agent import list_agent_classes
from rsi_loop.harness.config import RSILoopConfig, load_config
from tests.factories import make_run_plan


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
    assert prepared.command[:2] == ("/bin/bash", "-lc")
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


def test_prepare_maps_generic_reasoning_effort_to_codex_cli_override(
    tmp_path: Path,
) -> None:
    prompt_path = (tmp_path / "reasoning-prompt.md").resolve()
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"model": "gpt-test", "reasoning_effort": "xhigh"}
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
        "-lc",
        "codex exec -c 'model_reasoning_effort=\"xhigh\"' "
        '-c web_search="disabled" --model gpt-test '
        "--dangerously-bypass-approvals-and-sandbox "
        '"$(cat /tmp/rsi-agent-prompt.md)"',
    )


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

    assert prepared.command[:2] == ("/bin/bash", "-lc")
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
    assert redact_exact_values(
        "step 10, loss 0.01, provider-secret",
        rsi_loop_runtime_secret_values(config),
    ) == "step 10, loss 0.01, [REDACTED]"
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
        ("ANTHROPIC_AUTH_TOKEN", "provider-credential", True),
        ("OPENAI_API_KEY", "provider-credential", True),
        ("HF_TOKEN", "hub-credential", True),
        ("AWS_ACCESS_KEY_ID", "access-credential", True),
        ("AWS_SECRET_ACCESS_KEY", "aws-credential", True),
        ("AWS_SESSION_TOKEN", "session-credential", True),
        ("CLIENT_SECRET", "client-credential", True),
        ("CUSTOM_PASSWORD", "password-credential", True),
        ("CUSTOM_PRIVATE_KEY", "private-credential", True),
        ("custom_token", "1", True),
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
        endpoint, "proxy-user", "p%40ssword", "p@ssword",
    }


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
        writer.append(encoded[offset:offset + 7])
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


def test_prepare_rejects_unsupported_claude_reasoning_effort_before_writing_prompt(
    tmp_path: Path,
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
                            "reasoning_effort": "minimal",
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


@pytest.mark.parametrize("effort", ("none", "max", "HIGH", ""))
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


@pytest.mark.parametrize("agent_name,settings_path", [
    ("claude-code", "/home/agent/.claude/settings.json"),
    ("codex", "/etc/codex/hooks.json"),
])
@pytest.mark.parametrize("read_only_copy", [True, False])
def test_stop_hook_runs_after_read_only_or_mode_reset_copy(
    tmp_path: Path, read_only_copy: bool, agent_name: str, settings_path: str,
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
                argv, capture_output=True, text=True,
                env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
            )
            return AgentRunResult(
                exit_code=result.returncode, output=result.stdout + result.stderr,
            )

    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(update={"task": plan.task.model_copy(update={
        "agent": plan.task.agent.model_copy(update={"name": agent_name}),
    })})
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
        [str(hook)], input="{}", capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout) == {
        "decision": "block",
        "reason": "Do not stop. Continue working on the implementation.",
    }
    assert hook.stat().st_mode & 0o777 == 0o755
    assert not chmod_attempt.exists()


@pytest.mark.parametrize("agent_name,settings_filename", [
    ("claude-code", "_claude_settings.json"),
    ("codex", "_codex_hooks.json"),
])
@pytest.mark.parametrize("successful_checks", [0, 1])
def test_stop_hook_setup_failure_does_not_register_hook(
    tmp_path: Path, successful_checks: int, agent_name: str, settings_filename: str,
) -> None:
    class FailingRuntime(RecordingAgentRuntime):
        def exec(self, container, command, **kwargs):
            super().exec(container, command, **kwargs)
            return AgentRunResult(
                exit_code=0 if len(self.executions) <= successful_checks else 1,
                output="hook permissions unavailable",
            )

    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(update={"task": plan.task.model_copy(update={
        "agent": plan.task.agent.model_copy(update={"name": agent_name}),
    })})
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
