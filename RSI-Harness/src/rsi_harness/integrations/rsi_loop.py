"""Narrow integration with RSI Loop's public Agent registry and hooks."""

from __future__ import annotations

import json
import logging
import re
import shlex
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    PROVIDER_ERRORS,
    AgentHookRequest,
    AgentPrepareRequest,
    AgentResume,
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
from rsi_loop.harness.agent.oracle import SOLUTION_DIR, SUBMISSIONS_ENV
from rsi_loop.harness.backend import ExecResult
from rsi_loop.harness.config import RSILoopConfig

_CONTAINER_PROMPT_PATH = PurePosixPath("/tmp/rsi-agent-prompt.md")
# Next to agent_prompt.md: the next user turn of a continued session, which
# is all its CLI reads from the container prompt path.
_RESUME_PROMPT_NAME = "agent_resume_prompt.md"
_PROVIDER_PROMPT_PATH = PurePosixPath("/tmp/rsi-agent-provider-retry.md")
_PROVIDER_PROMPT = (
    "Your previous turn ended because the model service was temporarily "
    "{reason}. Continue the same task and session. The Harness has NOT restarted "
    "Work: background training and other jobs may still be running. Check their "
    "state before launching replacements. Existing submissions and the original "
    "time budget still apply.\n"
)
_PROVIDER_PROMPT_REASONS = {
    "at_capacity": "at capacity",
    "rate_limited": "rate limiting requests",
    "unavailable": "unavailable",
}
# Exponential backoff before each consecutive retry of a provider error; once
# they are spent, the Agent stops (the run can be resumed later). The waits
# run inside the Agent's exec, so they count against its time budget.
_PROVIDER_RETRY_DELAYS = (30, 60, 120, 240, 480, 900)
# An exec that worked this long before failing again starts the backoff over.
_PROVIDER_RETRY_RESET_SECONDS = 900
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")
_CODEX_ROLLOUT_DIAGNOSTIC = re.compile(
    r"\d{4}-\d\d-\d\dT\S+\s+ERROR codex_core::session: "
    r"failed to record rollout items:.*"
)
# How every Agent (its CLI, a driven CLI's driver, a resume, the oracle) is
# started in Work. Not a login shell: a login profile may reset PATH (Debian's
# and Alpine's /etc/profile do), dropping the image's own ENV PATH and the
# sandbox client directory the exec prepends. Nothing the Agent runs needs a
# profile: the CLIs, their node and rsi-submit are in /usr/local/bin, put
# there by the Work image (or, on a cluster, bound there).
AGENT_SHELL = ("/bin/bash", "-c")
_CONTROL_ENV_KEYS = ("RSI_JUDGE_URL", "RSI_TOKEN")
# Where install_hooks stages a task's solution/ for an Agent that runs it.
_SOLUTION_ARCHIVE_PATH = PurePosixPath("/tmp/rsi-oracle-solution.tar")


def task_solution_script(task_dir: Path) -> Path:
    """Return the task's ``solution/solve.sh``, or raise :class:`SetupError`."""

    script = Path(task_dir) / "solution" / "solve.sh"
    if script.is_symlink() or not script.is_file():
        raise SetupError(
            "the oracle runs solution/solve.sh, which the task does not have "
            "as a regular file"
        )
    return script


def _solution_archive(solution: Path, target: Path) -> None:
    """Write *solution*'s contents to a tar; symlinks stay symlinks."""

    def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as archive:
        for child in sorted(solution.iterdir()):
            archive.add(child, arcname=child.name, filter=anonymous)


def _solution_install_script() -> str:
    archive = shlex.quote(str(_SOLUTION_ARCHIVE_PATH))
    solution = shlex.quote(SOLUTION_DIR)
    # Owned by the Work user. rsi-oracle removes it after solve.sh, or only
    # empties it when its exec user cannot unlink it from the root-owned /.
    return (
        f"set -e; rm -rf {solution}; install -d -m 0755 {solution}; "
        f"tar -xf {archive} -C {solution} --no-same-owner; "
        f"chown -hR agent:agent {solution}; chmod u+x {solution}/solve.sh; "
        f"rm -f {archive}"
    )


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
        url = endpoint
        if endpoint in (config.http_proxy, config.https_proxy) and (
            "://" not in endpoint
        ):
            # A proxy without a scheme is read as http://, credentials too.
            url = f"http://{endpoint.strip()}"
        try:
            parsed = urlsplit(url)
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
    """Build the exact non-control environment passed to an RSI Loop Agent.

    An Agent that runs the task's own solution gets the proxy settings only:
    no credential, model, endpoint or extra variable reaches its Work.
    """

    environment: dict[str, str] = {}
    for value, lower_name, upper_name in (
        (config.http_proxy, "http_proxy", "HTTP_PROXY"),
        (config.https_proxy, "https_proxy", "HTTPS_PROXY"),
        (config.no_proxy, "no_proxy", "NO_PROXY"),
    ):
        if value:
            environment[lower_name] = value
            environment[upper_name] = value
    if getattr(agent, "runs_task_solution", False):
        return environment
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


def _run_session_id(plan: Any) -> str:
    """The id a run's first Agent session takes where the CLI lets the
    caller choose it."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, plan.paths.root.as_uri()))


class _ControlBinding:
    __slots__ = ("container", "submit_url", "token")

    def __init__(self, *, container: ContainerRef, submit_url: str, token: str) -> None:
        self.container = container
        self.submit_url = submit_url
        self.token = token


@dataclass(frozen=True)
class _ProviderRetry:
    command: tuple[str, ...]
    resume_command: Callable[[str], tuple[str, ...]]
    session_id_in: Callable[[str], str | None]


# Codex's terminal errors once its own retries are spent, e.g.
# "exceeded retry limit, last status: 429 Too Many Requests, request id: ..."
# or "unexpected status 503 Service Unavailable: ...".
_CODEX_RETRY_LIMIT = re.compile(r"exceeded retry limit, last status: (\d{3})\b.*")
_CODEX_SERVER_ERROR = re.compile(r"unexpected status 5\d\d\b.*", re.DOTALL)
# A 5xx body, e.g. a gateway's HTML page, can span lines. Only this much of the
# output's end is searched for it, so an old mention in tool output is not read
# as the terminal error.
_CODEX_ERROR_TAIL_CHARS = 2000
_CODEX_MESSAGES = {
    "Selected model is at capacity. Please try a different model.": "at_capacity",
    "Policy-restricted capacity is unavailable. Please try again later.": (
        "at_capacity"
    ),
    "Flex capacity unavailable.": "at_capacity",
    # HTTP 500.
    "We're currently experiencing high demand, which may cause temporary errors.": (
        "unavailable"
    ),
}
_CODEX_JSON_ERRORS = {
    "server_overloaded": "at_capacity",
    "rate_limit_exceeded": "rate_limited",
}


def _codex_provider_error(output: str) -> str | None:
    """The provider error (a PROVIDER_ERRORS key) Codex ended on: only its
    explicit terminal error, not mentions in tool output. Quota and usage
    limits are not retried: they do not lift within the backoff."""

    lines = _ANSI_SGR.sub("", output).splitlines()
    # Only skip the CLI's known footer/rollout shutdown diagnostics. An old
    # provider error inside tool output must not mask a different later failure.
    while lines:
        if not lines[-1].strip() or _CODEX_ROLLOUT_DIAGNOSTIC.fullmatch(lines[-1]):
            lines.pop()
        elif (
            len(lines) >= 2
            and re.fullmatch(r"[0-9][0-9,]*", lines[-1])
            and lines[-2] == "tokens used"
        ):
            del lines[-2:]
        else:
            break
    if not lines:
        return None
    if not lines[-1].startswith("ERROR: "):
        tail = ("\n" + "\n".join(lines))[-_CODEX_ERROR_TAIL_CHARS:]
        start = tail.rfind("\nERROR: ")
        server_error = start >= 0 and _CODEX_SERVER_ERROR.fullmatch(tail[start + 8 :])
        return "unavailable" if server_error else None
    # Newer Codex releases print a typographic apostrophe.
    error = lines[-1][7:].replace("\u2019", "'")
    if (kind := _CODEX_MESSAGES.get(error)) is not None:
        return kind
    if _CODEX_SERVER_ERROR.fullmatch(error) is not None:
        return "unavailable"
    if (limit := _CODEX_RETRY_LIMIT.fullmatch(error)) is not None:
        status = int(limit.group(1))
        if status == 429:
            return "rate_limited"
        return "unavailable" if 500 <= status < 600 else None
    try:
        detail = json.loads(error)
    except ValueError:
        return None
    if not isinstance(detail, dict):
        return None
    detail = detail.get("error", detail)
    if not isinstance(detail, dict):
        return None
    for key in ("code", "codex_error_info"):
        if (kind := _CODEX_JSON_ERRORS.get(str(detail.get(key)))) is not None:
            return kind
    return None


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
        self._provider_retry: _ProviderRetry | None = None

    def available_agents(self) -> tuple[str, ...]:
        return tuple(sorted(agent_class.name for agent_class in list_agent_classes()))

    def prepare(self, request: AgentPrepareRequest) -> PreparedAgent:
        self._provider_retry = None
        plan = request.run_plan
        agent_plan = plan.task.agent
        validate_agent_reasoning_effort(agent_plan.name, agent_plan.reasoning_effort)
        agent = validate_agent_config(
            self._config, agent_plan.name, model=agent_plan.model
        )
        model = agent_plan.model or self._config.agent_model or agent.default_model
        request.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        final_judge = plan.task.verifier.final_judge
        if request.resume is None or not request.prompt_path.exists():
            request.prompt_path.write_text(
                self._final_judge_prompt(
                    plan.task.instruction,
                    plan.gpu_plan,
                    (
                        request.resume.seconds_left
                        if request.resume is not None
                        and request.resume.seconds_left is not None
                        else agent_plan.timeout_seconds
                    ),
                    public_tests=plan.task.verifier.visibility == "public",
                    stop_hook=agent_plan.install_stop_hook,
                )
                if final_judge
                else self._prompt(
                    plan.task.instruction,
                    plan.gpu_plan,
                    request.max_submissions,
                    public_tests=plan.task.verifier.visibility == "public",
                )
            )
        # The first attempt's session takes the run's own id where the CLI
        # lets it choose one.
        prompt_path, session_id = request.prompt_path, _run_session_id(plan)
        if request.resume is not None:
            prompt_path, session_id = self._resume_prompt(
                agent, request.prompt_path, request.resume, final_judge=final_judge
            )
        try:
            command = agent.format_run_cmd(
                str(_CONTAINER_PROMPT_PATH),
                model=model,
                cwd=str(plan.workdir),
                internet=agent_plan.network.mode == "public",
                resume=request.resume is not None and session_id is not None,
                continuation=STOP_HOOK_REASON if agent_plan.install_stop_hook else None,
                session_id=session_id,
            )
        except ValueError as error:
            # A backstop: production checks the model before any image or
            # container work, but a refusal here must still be a setup error.
            raise SetupError(str(error)) from error
        command = _apply_reasoning_effort(agent, command, agent_plan.reasoning_effort)
        environment = self._agent_environment(agent, model)
        if getattr(agent, "runs_task_solution", False):
            # A final-judge run's one round is the Harness's, after solve.sh.
            environment[SUBMISSIONS_ENV] = (
                "0" if final_judge else str(request.max_submissions or 1)
            )
            # Explicitly override task-image/host values. Only a backend with
            # a native pre-Judge cleanup barrier may defer solution removal.
            environment["RSI_ORACLE_SOLUTION_CLEANUP"] = "launcher"
        prepared = PreparedAgent(
            agent_name=agent.name,
            command=(*AGENT_SHELL, command),
            prompt_path=prompt_path,
            environment=tuple(sorted(environment.items())),
        )
        if agent.name == "codex":

            def resume_command(session_id: str) -> tuple[str, ...]:
                command = agent.format_run_cmd(
                    str(_PROVIDER_PROMPT_PATH),
                    model=model,
                    cwd=str(plan.workdir),
                    internet=agent_plan.network.mode == "public",
                    resume=True,
                    session_id=session_id,
                )
                return (
                    *AGENT_SHELL,
                    _apply_reasoning_effort(
                        agent, command, agent_plan.reasoning_effort
                    ),
                )

            self._provider_retry = _ProviderRetry(
                prepared.command, resume_command, agent.session_id_in
            )
        return prepared

    def _resume_prompt(
        self,
        agent: Agent,
        prompt_path: Path,
        resume: AgentResume,
        *,
        final_judge: bool = False,
    ) -> tuple[Path, str | None]:
        """Add the attempt's message to the prompt; return what the CLI reads.

        The Agent continues the last session the run's output names with the
        message as its next user turn. Without one it starts a new session
        with the whole prompt, the message included. A final-judge attempt's
        message also gives its new deadline.
        """

        message = resume.message.rstrip() + "\n"
        if final_judge and resume.seconds_left is not None:
            message += (
                f"New deadline: {_deadline(resume.seconds_left)} (check with "
                "`date -u`); it replaces any earlier deadline.\n"
            )
        # Appended, never rewritten: Judge trajectory copies are byte ranges
        # of this file.
        with prompt_path.open("a") as prompt:
            prompt.write(f"\n## Attempt {resume.attempt}\n\n{message}")
        if not agent.resumes_sessions:
            return prompt_path, None
        session_id = agent.last_session_id(resume.agent_output)
        if session_id is None:
            self._logger.warning(
                "Agent %s: no session of an earlier attempt in %s; attempt %d "
                "starts a new session with the whole prompt",
                agent.name,
                resume.agent_output,
                resume.attempt,
            )
            return prompt_path, None
        turn_path = prompt_path.with_name(_RESUME_PROMPT_NAME)
        turn_path.write_text(message)
        return turn_path, session_id

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
        if getattr(agent, "runs_task_solution", False):
            self._stage_solution(runtime, request)
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
            result = self._exec_with_provider_retry(request, runtime, exec_options)
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

    def _exec_with_provider_retry(
        self,
        request: AgentRunRequest,
        runtime: AgentRuntime,
        exec_options: dict[str, Any],
    ) -> AgentRunResult:
        retry = self._provider_retry
        command = request.prepared.command
        if (
            retry is None
            or request.prepared.agent_name != "codex"
            or command != retry.command
        ):
            return runtime.exec(request.container, command, **exec_options)

        deadline = (
            None
            if request.timeout_seconds is None
            else time.monotonic() + request.timeout_seconds
        )
        outputs: list[str] = []
        captured = True
        truncated = False
        started = False
        session_id: str | None = None
        retries = consecutive = 0
        waited = delay = 0.0
        stopped: str | None = None
        while True:
            exec_started_at = time.monotonic()
            result = runtime.exec(request.container, command, **exec_options)
            outputs.append(result.output)
            captured = captured and result.full_output_captured
            truncated = truncated or result.output_truncated
            started = started or result.exec_started
            # Use this exec's header, not --last or an old session printed by a
            # tool reading a previous trajectory. Bounded runtime output keeps
            # the header as well as the terminal error.
            if session_id is None:
                session_id = next(
                    (
                        found
                        for line in result.output.splitlines()[:64]
                        if (found := retry.session_id_in(line)) is not None
                    ),
                    None,
                )
            kind = (
                _codex_provider_error(result.output)
                if result.exit_code == 1
                and result.exec_started
                and not result.timed_out
                and not result.cancelled
                else None
            )
            if kind is None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                result = replace(result, exit_code=None, timed_out=True)
                break
            if time.monotonic() - exec_started_at - delay >= (
                _PROVIDER_RETRY_RESET_SECONDS
            ):
                consecutive = 0
            delay = _PROVIDER_RETRY_DELAYS[
                min(consecutive, len(_PROVIDER_RETRY_DELAYS) - 1)
            ]
            remaining = None if deadline is None else deadline - time.monotonic()
            if (
                session_id is None
                or consecutive == len(_PROVIDER_RETRY_DELAYS)
                or (remaining is not None and remaining <= delay)
            ):
                # Stopping keeps the rest of the budget for a later resume.
                stopped = kind
                self._logger.warning(
                    "RSI Harness: Codex's model provider %s; the Agent stops after %d "
                    "retries. Resume the run once the provider recovers.",
                    PROVIDER_ERRORS[kind],
                    retries,
                )
                break
            with tempfile.TemporaryDirectory(prefix="rsi-provider-retry-") as raw:
                prompt = Path(raw) / "prompt.md"
                prompt.write_text(
                    _PROVIDER_PROMPT.format(reason=_PROVIDER_PROMPT_REASONS[kind])
                )
                runtime.copy_to(request.container, prompt, _PROVIDER_PROMPT_PATH)
            retries += 1
            consecutive += 1
            waited += delay
            notice = (
                f"RSI Harness: Codex's model provider {PROVIDER_ERRORS[kind]}; retry "
                f"{retries} in {delay}s in the same session. Work is still running."
            )
            self._logger.warning("%s", notice)
            resumed = retry.resume_command(session_id)
            # Waiting inside the bounded exec keeps cancellation/timeout in the
            # runtime's existing containment path, with Work and GPU jobs alive.
            command = (
                *resumed[:2],
                f"printf '%s\\n' {shlex.quote(notice)}\nsleep {delay} &&\n{resumed[2]}",
            )
            if deadline is not None:
                # Measured after the copy, so the retry still ends by the deadline.
                exec_options["timeout_seconds"] = deadline - time.monotonic()
            # A sandbox's Work deadline must only be activated once, never
            # reset by a provider retry. The remaining timeout includes sleep.
            exec_options.pop("on_exec_start", None)

        return replace(
            result,
            output="\n".join(outputs),
            exec_started=started,
            full_output_captured=captured,
            output_truncated=truncated,
            provider_error=stopped,
            provider_retries=retries,
            provider_wait_seconds=waited,
        )

    @staticmethod
    def _stage_solution(runtime: AgentRuntime, request: AgentHookRequest) -> None:
        """Install the task's ``solution/`` at ``/solution`` in Work, as root.

        Before the Agent's exec, so a sandbox's Work deadline still starts
        when the oracle does.
        """

        task_solution_script(request.run_plan.task.source_dir)
        solution = request.run_plan.task.source_dir / "solution"
        with tempfile.TemporaryDirectory(prefix="rsi-oracle-") as raw:
            archive = Path(raw) / "solution.tar"
            _solution_archive(solution, archive)
            runtime.copy_to(request.container, archive, _SOLUTION_ARCHIVE_PATH)
        result = runtime.exec(
            request.container,
            ("/bin/sh", "-c", _solution_install_script()),
            user="root",
        )
        if result.exit_code != 0:
            raise InfrastructureError(
                f"staging the task solution in Work failed: {result.output}"
            )

    def clear_transient_bindings(self) -> None:
        """Drop runtime-only submission authority on every cleanup path."""

        self._control = None
        self._provider_retry = None

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

    @staticmethod
    def _final_judge_prompt(
        instruction: str,
        gpu_plan: RunGPUPlan,
        seconds: float,
        *,
        public_tests: bool = False,
        stop_hook: bool = True,
    ) -> str:
        """The Evaluation section of a run with one final judge."""
        tests_guidance = (
            "The Judge's tests are public: `/tests` holds, read-only, exactly the\n"
            "files the Judge runs as `/tests/test.sh`.\n"
            if public_tests
            else ""
        )
        gpu_guidance = (
            "Every Work GPU process must exit before rsi-submit; a rejected\n"
            "preflight does not use up your one call.\n"
            if gpu_plan.judge_mode is JudgeGPUMode.RELEASE_ALL
            else ""
        )
        stop_guidance = (
            ""
            if stop_hook
            else "If you stop earlier, the Judge evaluates the WORKDIR when you stop.\n"
        )
        budget = (
            f"Time budget: {_duration(seconds)} of wall-clock time; "
            f"deadline {_deadline(seconds)}"
        )
        return f"""{instruction.rstrip()}

## Evaluation

This run has exactly one final judge. It evaluates the entire WORKDIR once,
and that one result is the score of the run.
You may call `rsi-submit` once to be judged early: the Judge then evaluates
the WORKDIR as it is at that moment, you receive the result, and the run
ends. Otherwise the Judge evaluates the WORKDIR automatically when your
time ends.
{stop_guidance}{budget}
(check with `date -u`). Work stops at the deadline, so keep the WORKDIR
in a state you want judged.
Checks you run yourself score nothing; only the final judge does.
{tests_guidance}{gpu_guidance}"""


def _deadline(seconds: float) -> str:
    """When an Agent budget that starts now ends, in UTC to the minute."""
    return f"{datetime.now(UTC) + timedelta(seconds=seconds):%Y-%m-%d %H:%M} UTC"


def _duration(seconds: float) -> str:
    """6 h, 90 min or 45 s: the largest unit that states the budget exactly."""
    whole = round(seconds)
    if whole and whole % 3600 == 0:
        return f"{whole // 3600} h"
    if whole and whole % 60 == 0:
        return f"{whole // 60} min"
    return f"{whole} s"


__all__ = [
    "AGENT_SHELL",
    "RSILoopAgentAdapter",
    "RSILoopBackendBridge",
    "rsi_loop_agent_environment",
    "rsi_loop_oneshot",
    "rsi_loop_runtime_secret_values",
    "task_solution_script",
    "validate_agent_config",
    "validate_agent_reasoning_effort",
]
