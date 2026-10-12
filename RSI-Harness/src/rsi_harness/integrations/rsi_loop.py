"""Narrow integration with RSI Loop's public Agent registry and hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    JudgeGPUMode,
    PreparedAgent,
    RunGPUPlan,
)
from rsi_harness.runtime.redaction import (
    is_credential_name,
    redact_exact_values,
    redact_text,
)
from rsi_loop.harness.agent import (
    Agent,
    create_agent,
    get_agent_class,
    list_agent_classes,
)
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.backend import ExecResult
from rsi_loop.harness.config import RSILoopConfig

_CONTAINER_PROMPT_PATH = PurePosixPath("/tmp/rsi-agent-prompt.md")
_CONTROL_ENV_KEYS = ("RSI_JUDGE_URL", "RSI_TOKEN")


def validate_agent_reasoning_effort(
    agent_name: str, reasoning_effort: str | None
) -> None:
    """Fail before runtime mutation when an Agent cannot honor the option."""

    if reasoning_effort is None:
        return
    try:
        agent_class = get_agent_class(agent_name)
    except ValueError:
        agent_class = None
    if agent_class is None or not agent_class.reasoning_efforts:
        raise SetupError(f"Agent {agent_name!r} does not support reasoning effort")
    if reasoning_effort not in agent_class.reasoning_efforts:
        supported = ", ".join(sorted(agent_class.reasoning_efforts))
        display_name = agent_class.display_name or agent_class.name
        raise SetupError(
            f"unsupported {display_name} reasoning effort {reasoning_effort!r}; "
            f"choose one of: {supported}"
        )


def validate_agent_config(
    config: RSILoopConfig, agent_name: str, *, model: str | None = None
) -> Agent:
    """Fail before runtime mutation when the configuration cannot run the Agent.

    *model* is the run's own model (request or task), if any; the configured
    and the Agent's default model apply without it. Returns the checked Agent
    so callers need not build it again.
    """

    try:
        agent = create_agent(agent_name, config)
        agent.check_config()
    except ValueError as error:
        raise SetupError(str(error)) from error
    if agent.requires_model and not (
        model or config.agent_model or agent.default_model
    ):
        raise SetupError(
            f"Agent {agent.name!r} needs a model; set RSI_AGENT_MODEL or --model"
        )
    return agent


def _apply_reasoning_effort(
    agent: Agent, command: str, reasoning_effort: str | None
) -> str:
    validate_agent_reasoning_effort(agent.name, reasoning_effort)
    if reasoning_effort is None:
        return command
    try:
        return agent.apply_reasoning_effort(command, reasoning_effort)
    except ValueError as error:
        raise SetupError(str(error)) from error


def rsi_loop_runtime_secret_values(config: RSILoopConfig) -> set[str]:
    """Return exact RSI Loop runtime values that must never reach artifacts."""

    secret_names = set(config.agent_secret_env_names)
    values = {
        config.agent_api_key or "",
        *(
            str(value)
            for name, value in config.agent_extra_env.items()
            if name in secret_names or is_credential_name(name)
        ),
    }
    # EXTRA_ENV also carries ordinary flags, counts and paths. Globally
    # replacing their values (especially "0"/"1") corrupts the trajectory.
    for endpoint in (
        config.http_proxy,
        config.https_proxy,
        config.agent_api_base_url,
        *(
            str(value)
            for value in config.agent_extra_env.values()
            if "://" in str(value)
        ),
    ):
        if not endpoint:
            continue
        try:
            parsed = urlsplit(endpoint)
        except ValueError:
            # An arbitrary setting resembling a URL is not necessarily one.
            continue
        if parsed.username is not None or parsed.password is not None:
            values.add(endpoint)
            for credential in (parsed.username, parsed.password):
                if credential:
                    values.add(credential)
                    values.add(unquote(credential))
    return {value for value in values if value}


def rsi_loop_agent_environment(
    config: RSILoopConfig,
    agent: Agent,
    model: str | None,
) -> dict[str, str]:
    """Build the exact non-control environment passed to an RSI Loop Agent."""

    environment: dict[str, str] = {}
    for value, lower_name, upper_name in (
        (config.http_proxy, "http_proxy", "HTTP_PROXY"),
        (config.https_proxy, "https_proxy", "HTTPS_PROXY"),
        (config.no_proxy, "no_proxy", "NO_PROXY"),
    ):
        if value:
            environment[lower_name] = value
            environment[upper_name] = value
    if config.agent_api_key:
        environment[agent.api_key_env] = config.agent_api_key
    if config.agent_api_base_url and agent.api_base_env:
        environment[agent.api_base_env] = config.agent_api_base_url
    if model and agent.model_env:
        environment[agent.model_env] = model
    if config.nodejs_mirror_url:
        environment["RSI_NODEJS_MIRROR_URL"] = config.nodejs_mirror_url
    if config.npm_registry_url:
        environment["npm_config_registry"] = config.npm_registry_url
    for value, name in (
        (config.agent_context_window, "RSI_AGENT_CONTEXT_WINDOW"),
        (config.agent_max_output_tokens, "RSI_AGENT_MAX_OUTPUT_TOKENS"),
    ):
        if value:
            environment[name] = value
    environment.update(config.agent_extra_env)
    agent.augment_env(environment, model)
    for key in _CONTROL_ENV_KEYS:
        environment.pop(key, None)
    return environment


def rsi_loop_oneshot(config: RSILoopConfig, agent_plan: Any) -> dict[str, Any]:
    """The run's Agent as one non-interactive call, for ``rsi-agent`` in a Judge.

    The same CLI flags, environment (key, base URL, proxy, model) and
    reasoning effort as the Work Agent. ``command`` is ``None`` when the Agent
    has no one-shot command; ``unsupported`` then says why.
    """

    validate_agent_reasoning_effort(agent_plan.name, agent_plan.reasoning_effort)
    agent = create_agent(agent_plan.name, config)
    model = agent_plan.model or config.agent_model or agent.default_model
    result: dict[str, Any] = {
        "agent": agent.name,
        "model": model,
        "reasoning_effort": agent_plan.reasoning_effort,
        "environment": rsi_loop_agent_environment(config, agent, model),
        "command": None,
        "schema_option": None,
        "output": None,
        "unsupported": None,
    }
    try:
        oneshot = agent.format_oneshot_cmd(
            model=model, internet=agent_plan.network.mode == "public"
        )
    except ValueError as error:
        raise SetupError(str(error)) from error
    if oneshot is None:
        result["unsupported"] = (
            f"the run's Agent {agent.name!r} cannot run one-shot in a Judge"
        )
        return result
    result["command"] = _apply_reasoning_effort(
        agent, oneshot.command, agent_plan.reasoning_effort
    )
    result["schema_option"] = oneshot.schema_option
    result["output"] = {
        "mode": oneshot.output,
        "fields": list(oneshot.output_fields),
        "error_field": oneshot.error_field,
    }
    return result


class _ControlBinding:
    __slots__ = ("container", "submit_url", "token")

    def __init__(self, *, container: ContainerRef, submit_url: str, token: str) -> None:
        self.container = container
        self.submit_url = submit_url
        self.token = token


class AgentRuntime(Protocol):
    """Engine operations needed by the Agent integration."""

    def copy_to(
        self, container: ContainerRef, source: Path, target: PurePosixPath
    ) -> None: ...

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
        on_exec_start: Callable[[float], None] | None = None,
    ) -> AgentRunResult: ...


class RSILoopContainerHandle:
    """Present an Engine container reference to an RSI Loop Agent hook."""

    def __init__(self, container: ContainerRef) -> None:
        self.container = container

    @property
    def id(self) -> str:
        return self.container.container_id

    @property
    def name(self) -> str:
        return self.container.container_id

    @property
    def ip_address(self) -> None:
        return None


class RSILoopBackendBridge:
    """Translate only the file-copy and exec calls used by stop hooks."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime

    def copy_to_container(
        self,
        handle: RSILoopContainerHandle,
        src: Path,
        dst: PurePosixPath,
    ) -> None:
        self._runtime.copy_to(handle.container, src, dst)

    def exec_run(
        self,
        handle: RSILoopContainerHandle,
        cmd: str | list[str],
        *,
        user: str | None = None,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
        detach: bool = False,
    ) -> ExecResult:
        if workdir is not None or detach:
            raise ValueError("RSI Loop stop hooks requested unsupported exec options")
        result = self._runtime.exec(
            handle.container,
            cmd,
            user=user,
            environment=environment,
        )
        return ExecResult(
            output=result.output,
            exit_code=1 if result.exit_code is None else result.exit_code,
        )


class RSILoopAgentAdapter:
    """Use RSI Loop for Agent discovery, formatting, and stop hooks only."""

    def __init__(
        self,
        config: RSILoopConfig,
        *,
        runtime: AgentRuntime | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._logger = logger or logging.getLogger(__name__)
        self._control: _ControlBinding | None = None

    def available_agents(self) -> tuple[str, ...]:
        return tuple(sorted(agent_class.name for agent_class in list_agent_classes()))

    def prepare(self, request: AgentPrepareRequest) -> PreparedAgent:
        plan = request.run_plan
        agent_plan = plan.task.agent
        validate_agent_reasoning_effort(agent_plan.name, agent_plan.reasoning_effort)
        agent = validate_agent_config(
            self._config, agent_plan.name, model=agent_plan.model
        )
        model = agent_plan.model or self._config.agent_model or agent.default_model
        prompt = self._prompt(
            plan.task.instruction,
            plan.gpu_plan,
            request.max_submissions,
            public_tests=plan.task.verifier.visibility == "public",
        )
        request.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        request.prompt_path.write_text(prompt)
        try:
            command = agent.format_run_cmd(
                str(_CONTAINER_PROMPT_PATH),
                model=model,
                cwd=str(plan.workdir),
                internet=agent_plan.network.mode == "public",
                resume=request.resume,
                continuation=STOP_HOOK_REASON if agent_plan.install_stop_hook else None,
            )
        except ValueError as error:
            # A backstop: production checks the model before any image or
            # container work, but a refusal here must still be a setup error.
            raise SetupError(str(error)) from error
        command = _apply_reasoning_effort(agent, command, agent_plan.reasoning_effort)
        return PreparedAgent(
            agent_name=agent.name,
            command=("/bin/bash", "-lc", command),
            prompt_path=request.prompt_path,
            environment=tuple(sorted(self._agent_environment(agent, model).items())),
        )

    def install_hooks(self, request: AgentHookRequest) -> None:
        if request.container.role != "work":
            raise InfrastructureError(
                "Agent control environment requires a Work container"
            )
        if not request.submit_url or not request.token:
            raise InfrastructureError("Agent control environment is incomplete")
        if self._control is not None:
            raise InfrastructureError("Agent control environment is already installed")
        runtime = self._require_runtime()
        agent = create_agent(request.run_plan.task.agent.name, self._config)
        if request.run_plan.task.agent.install_stop_hook:
            log_dir = request.run_plan.paths.logs
            log_dir.mkdir(parents=True, exist_ok=True)
            agent.install_stop_hook(
                RSILoopBackendBridge(runtime),  # type: ignore[arg-type]
                RSILoopContainerHandle(request.container),  # type: ignore[arg-type]
                log_dir,
                self._logger,
            )
        self._control = _ControlBinding(
            container=request.container,
            submit_url=request.submit_url,
            token=request.token,
        )

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        control = self._control
        try:
            if control is None:
                raise InfrastructureError("Agent control environment is unavailable")
            if request.container != control.container:
                raise InfrastructureError(
                    "Agent control environment is bound to a different Work container"
                )
            runtime = self._require_runtime()
            runtime.copy_to(
                request.container,
                request.prepared.prompt_path,
                _CONTAINER_PROMPT_PATH,
            )
            environment = dict(request.prepared.environment)
            environment.update(
                {
                    "RSI_JUDGE_URL": control.submit_url,
                    "RSI_TOKEN": control.token,
                }
            )
            output_redact_values = {
                *request.output_redact_values,
                control.submit_url,
                control.token,
                *rsi_loop_runtime_secret_values(self._config),
            }
            exec_options: dict[str, object] = {
                "timeout_seconds": request.timeout_seconds,
                "environment": environment,
            }
            if request.output_path is not None:
                exec_options["output_path"] = request.output_path
            if request.output_path is not None or request.output_callback is not None:
                exec_options["output_redact_values"] = tuple(
                    sorted(output_redact_values)
                )
            if request.output_callback is not None:
                exec_options["output_callback"] = request.output_callback
            if request.on_exec_start is not None:
                exec_options["on_exec_start"] = request.on_exec_start
            result = runtime.exec(
                request.container,
                request.prepared.command,
                **exec_options,
            )
            safe_output = redact_exact_values(
                result.output,
                {
                    control.submit_url,
                    control.token,
                    *rsi_loop_runtime_secret_values(self._config),
                },
            )
            safe_output = redact_text(safe_output)
            return replace(result, output=safe_output)
        except Exception as error:
            secret_values = set(rsi_loop_runtime_secret_values(self._config))
            if control is not None:
                secret_values.update((control.submit_url, control.token))
            safe_error = redact_text(redact_exact_values(str(error), secret_values))
            raise InfrastructureError(safe_error or type(error).__name__) from error
        finally:
            self.clear_transient_bindings()

    def clear_transient_bindings(self) -> None:
        """Drop runtime-only submission authority on every cleanup path."""

        self._control = None

    def _require_runtime(self) -> AgentRuntime:
        if self._runtime is None:
            raise RuntimeError(
                "an Engine Agent runtime is required for hooks and execution"
            )
        return self._runtime

    def _agent_environment(self, agent: Agent, model: str | None) -> dict[str, str]:
        return rsi_loop_agent_environment(self._config, agent, model)

    @staticmethod
    def _prompt(
        instruction: str,
        gpu_plan: RunGPUPlan,
        max_submissions: int | None,
        *,
        public_tests: bool = False,
    ) -> str:
        judge_gpu_count = len(gpu_plan.judge.devices)
        submission_guidance = (
            "Judge submissions are unlimited during this run."
            if max_submissions is None
            else "You may submit to the Judge at most "
            f"{max_submissions} times during this run."
        )
        if gpu_plan.judge_mode is JudgeGPUMode.FREEZE_ONLY:
            gpu_guidance = (
                "Work is paused during evaluation; GPU release is not required."
            )
        elif gpu_plan.judge_mode is JudgeGPUMode.DISJOINT:
            gpu_label = "GPU" if judge_gpu_count == 1 else "GPUs"
            gpu_guidance = (
                f"Judge uses {judge_gpu_count} separate {gpu_label}; "
                "Work GPU release is not required."
            )
        else:
            gpu_guidance = (
                "Every Work GPU process must exit before rsi-submit. A rejected "
                "preflight does not consume a submission."
            )
        tests_guidance = (
            "\nThe Judge's tests are public: `/tests` holds, read-only, exactly the\n"
            "files the Judge runs as `/tests/test.sh`.\n"
            if public_tests
            else ""
        )
        return f"""{instruction.rstrip()}

## Evaluation

The Judge evaluates the entire current WORKDIR as it exists when you submit.
The best valid primary score wins. You may call `rsi-submit` repeatedly to
receive feedback while improving the same workspace.
`rsi-submit --list` shows previous submissions; `rsi-submit --help` shows local usage.
Each submission stores complete Judge stdout and stderr in
`/run/rsi-harness/feedback/agent-N.log`, where N is the submission number.
{submission_guidance}
{tests_guidance}
{gpu_guidance}
"""


__all__ = [
    "RSILoopAgentAdapter",
    "RSILoopBackendBridge",
    "rsi_loop_agent_environment",
    "rsi_loop_oneshot",
    "rsi_loop_runtime_secret_values",
    "validate_agent_config",
    "validate_agent_reasoning_effort",
]
