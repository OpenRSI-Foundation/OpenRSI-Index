from __future__ import annotations

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
        (False, 'claude --effort max -p "$(cat /tmp/rsi-agent-prompt.md)"'),
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
    assert any("chmod a+x" in str(call["command"]) for call in runtime.executions)
    assert all(
        "runtime-only-token" not in source.read_text()
        for _, source, _ in runtime.copies
    )


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
