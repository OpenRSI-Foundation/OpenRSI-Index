# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract base class for agents."""

from __future__ import annotations

import abc
import json
import logging
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rsi_loop.harness.agent import versions
from rsi_loop.harness.backend import ContainerBackend, ContainerHandle
from rsi_loop.harness.config import RSILoopConfig

# What every stop mechanism (native hook, extension or driver) sends the
# Agent when it tries to end its work.
STOP_HOOK_REASON = "Do not stop. Continue working on the implementation."
CONTEXT_WINDOW_ENV = "RSI_AGENT_CONTEXT_WINDOW"
MAX_OUTPUT_TOKENS_ENV = "RSI_AGENT_MAX_OUTPUT_TOKENS"


def select_arch(what: str, *, x86_64: str, aarch64: str) -> str:
    """Return a bash step that sets ``arch`` to the image's release name.

    Install commands download native releases for the architecture the image
    runs on: ``uname -m`` inside the build. ``x86_64`` and ``aarch64`` (``arm64``
    on some systems) map to the names *what*'s upstream publishes them under; any
    other machine fails the build with a clear message instead of installing
    a binary that cannot run.
    """
    return (
        'case "$(uname -m)" in'
        f" x86_64) arch={x86_64} ;;"
        f" aarch64|arm64) arch={aarch64} ;;"
        f' *) echo "{what}: no release for architecture $(uname -m);'
        ' x86_64 and aarch64 are supported" >&2; exit 1 ;;'
        " esac"
    )


# What a session id read back from an Agent's output may look like before it
# goes into a command line or a protocol request.
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")


def link_session_store(link: str, store: str) -> str:
    """Return a bash step that keeps a CLI's session directory on the rootfs.

    ``--agent-auth local`` mounts a private tmpfs over the CLI's directory
    under ``$HOME`` for its login, and whatever else the CLI writes there is
    gone when Work stops: no retained image has it, so a later attempt could
    not continue the session. *link* (relative to ``$HOME``) becomes a symlink
    to *store* (likewise), a directory on the rootfs; the login stays on the
    tmpfs. An existing *link*, the symlink of an earlier attempt or the CLI's
    own directory, is kept.
    """
    link_path = f'"$HOME/{link}"'
    store_path = f'"$HOME/{store}"'
    return (
        f'mkdir -p {store_path} "$HOME/{PurePosixPath(link).parent}" && '
        f"{{ [ -e {link_path} ] || ln -sfn {store_path} {link_path}; }}"
    )


# Where NODE_INSTALL_CMD puts the pinned node. Agents run node by this path,
# never by name: a task image may put its own node first on PATH.
NODE_BIN = "/usr/local/bin/node"

# The pinned node every Agent image installs into /usr/local; the driver, its
# launcher wrapper and the ZCode launcher rely on it being there. A mirror
# must serve the same v<version>/node-v<version>-linux-<x64|arm64>.tar.xz
# layout as nodejs.org.
NODE_INSTALL_CMD = "sudo -E bash -c " + shlex.quote(
    "set -euo pipefail"
    f" && {select_arch('node', x86_64='x64', aarch64='arm64')}"
    " && NODE_MIRROR=${RSI_NODEJS_MIRROR_URL:-https://nodejs.org/dist}"
    f' && curl -fsSL "$NODE_MIRROR/v{versions.NODE_VERSION}'
    f'/node-v{versions.NODE_VERSION}-linux-${{arch}}.tar.xz"'
    " | tar -xJ -C /usr/local --strip-components=1"
)


@dataclass(frozen=True, slots=True)
class OneShotCommand:
    """A non-interactive Agent invocation that answers one prompt and exits.

    ``command`` is a bash command that reads the prompt on stdin. It may use
    two literal tokens: ``{output_file}``, the file the Agent must write its
    final message to (``output == "file"``), and ``{schema_option}``, replaced
    by ``schema_option`` (with ``{schema_file}`` filled in) when the caller
    asks for structured output, and by nothing otherwise. An Agent without a
    native schema option leaves ``schema_option`` as ``None``; the caller then
    states the schema in the prompt instead.

    With ``output == "stdout-json"`` the Agent prints one JSON object: the
    first of ``output_fields`` present is the final message, and a true
    ``error_field`` means the call failed.
    """

    command: str
    schema_option: str | None = None
    output: str = "file"
    output_fields: tuple[str, ...] = ()
    error_field: str | None = None

    def __post_init__(self) -> None:
        if self.output not in {"file", "stdout-json"}:
            raise ValueError(f"unsupported one-shot output {self.output!r}")
        if self.output == "file" and "{output_file}" not in self.command:
            raise ValueError("a file one-shot command must name {output_file}")
        if self.output == "stdout-json" and not self.output_fields:
            raise ValueError("a stdout-json one-shot command needs output fields")
        if "{schema_option}" not in self.command:
            raise ValueError("a one-shot command must place {schema_option}")


class Agent(abc.ABC):
    """Base class for coding agents.

    Subclasses declare agent configuration as class-level attributes
    (``name``, ``install_cmds``, ``run_cmd``, …) and override hook
    methods for agent-specific behavior.
    """

    # -- required (no default) --
    name: str
    install_cmds: list[str]
    run_cmd: str  # template: {prompt_file}, {cwd} placeholders
    api_key_env: str  # env var name the agent expects for API key

    # -- optional --
    api_base_env: str | None = None
    default_api_base_url: str | None = None
    model_env: str | None = None
    default_model: str | None = None
    # The CLI has no model of its own to fall back on: setup refuses a run
    # that names none.
    requires_model: bool = False
    # The CLI must be told the model's context window and output limit.
    requires_context_limits: bool = False
    timeout: int = 3600
    stop_hook: str | None = None
    resume_cmd: str | None = None
    # format_run_cmd continues a session of an earlier attempt, named by
    # last_session_id, with the prompt file as its next user turn.
    resumes_sessions: bool = False
    # (link, store) under $HOME as link_session_store keeps the CLI's
    # sessions off its login tmpfs; None when the CLI needs none.
    session_store: tuple[str, str] | None = None
    display_name: str | None = None
    # Values accepted by ``apply_reasoning_effort``; empty means the Agent
    # does not support a reasoning effort.
    reasoning_efforts: frozenset[str] = frozenset()
    # Files baked into the Work image before ``install_cmds`` run:
    # container path -> (resource path relative to the ``rsi_loop`` package,
    # file mode). Agents without any keep their Work image tag unchanged.
    image_files: Mapping[str, tuple[str, int]] = {}
    # Commands besides the ``run_cmd`` launcher that ``install_cmds`` provide
    # and the Agent starts. The Work image moves a task image's copy aside
    # and checks the installed one, as it does for the launcher.
    extra_launchers: tuple[str, ...] = ()
    # Not a model Agent: it runs the task's own solution/solve.sh in Work.
    # The Harness stages the solution for it and passes it no credential,
    # model or provider endpoint.
    runs_task_solution: bool = False

    def __init__(self, config: RSILoopConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        """Mutate *env* with agent-specific environment variables.

        Called **after** the generic env vars (proxy, API key, model,
        mirrors, extra_env) have already been applied.
        """

    def build_args(self) -> dict[str, str | None]:
        """Return the build arguments ``install_cmds`` read, by name.

        A Work image build declares every name and passes the values that
        are set; None leaves the install command's default. The values stay
        out of the image's tag, so they may only choose where verified
        content comes from.
        """
        return {}

    def format_run_cmd(
        self,
        prompt_path: str,
        *,
        model: str | None = None,
        cwd: str = "",
        internet: bool = True,
        resume: bool = False,
        continuation: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Build the shell command to run the agent.

        The default implementation expands ``{prompt_file}`` / ``{cwd}``
        placeholders and handles the resume fallback.  Subclasses that need
        extra CLI flags (``--model``, ``--disallowedTools``, …) should call
        ``super()`` first, then append.

        *continuation* is the text to send whenever the Agent ends a turn, or
        ``None`` when the stop hook is disabled. Agents whose stop hook is a
        native hook installed by :meth:`install_stop_hook` ignore it.

        *session_id* names the Agent's session. With *resume* it is the
        session of an earlier attempt to continue (``resumes_sessions``
        Agents only), and the prompt file holds its next user turn; without,
        the id a new session takes where the CLI lets the caller choose it.
        *resume* without a session id is ``resume_cmd``: the most recent
        session, with a fixed prompt.
        """
        if resume and session_id is None and self.resume_cmd:
            return self.resume_cmd
        return self.run_cmd.format(prompt_file=prompt_path, cwd=cwd)

    def last_session_id(self, output: Path) -> str | None:
        """Return the id of the last session this Agent started in *output*.

        *output* is the run's ``agent_output.txt``, which holds every attempt.
        ``None`` when it names none, or the last one is not a plain id.
        """
        found = None
        try:
            with output.open(encoding="utf-8", errors="replace") as lines:
                for line in lines:
                    found = self.session_id_in(line) or found
        except OSError:
            return None
        if found is None or _SESSION_ID.fullmatch(found) is None:
            return None
        return found

    def session_id_in(self, line: str) -> str | None:
        """Return the session id one line of the Agent's output names, if any.

        The default reads the ``session_id`` of a stream-json
        ``system``/``init`` line, which ``claude -p``, rsi-agent-driver and the
        pi extension print whenever a session starts or is continued.
        """
        if '"init"' not in line:
            return None
        try:
            event = json.loads(line)
        except ValueError:
            return None
        if (
            not isinstance(event, dict)
            or event.get("type") != "system"
            or event.get("subtype") != "init"
            or not isinstance(event.get("session_id"), str)
        ):
            return None
        return event["session_id"]

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand | None:
        """Build the one-shot command a Judge runs through ``rsi-agent``.

        The same CLI, credentials and provider settings as :meth:`format_run_cmd`,
        but non-interactive: one prompt in, one final message out, with no
        session, hooks or project instructions. ``None`` (the default) means
        this Agent cannot be called from a Judge.
        """
        del model, internet
        return None

    def check_config(self) -> None:
        """Raise :class:`ValueError` when the configuration cannot run this Agent.

        Called before the Agent is prepared, so a missing setting fails the
        run during setup instead of inside Work.
        """
        if self.requires_context_limits:
            self.context_limits()

    def context_limits(self) -> tuple[int, int]:
        """Return (context window, max output tokens) from the configuration.

        Raises :class:`ValueError` when either is missing or invalid. They are
        never guessed: a declared window above the real one makes some CLIs
        loop on failed requests instead of compacting.
        """
        values = {
            CONTEXT_WINDOW_ENV: self._config.agent_context_window,
            MAX_OUTPUT_TOKENS_ENV: self._config.agent_max_output_tokens,
        }
        parsed: dict[str, int] = {}
        for env_name, raw in values.items():
            text = "" if raw is None else str(raw).strip()
            if not text.isdecimal() or int(text) <= 0:
                state = "is not set" if not text else f"is {text!r}"
                raise ValueError(
                    f"Agent {self.name!r} needs {CONTEXT_WINDOW_ENV} and "
                    f"{MAX_OUTPUT_TOKENS_ENV} set to the model's real limits as "
                    f"positive token counts; {env_name} {state}"
                )
            parsed[env_name] = int(text)
        window = parsed[CONTEXT_WINDOW_ENV]
        output = parsed[MAX_OUTPUT_TOKENS_ENV]
        if output >= window:
            raise ValueError(
                f"{MAX_OUTPUT_TOKENS_ENV} ({output}) must be smaller than "
                f"{CONTEXT_WINDOW_ENV} ({window})"
            )
        return window, output

    def require_api_endpoint(self) -> None:
        """Raise :class:`ValueError` unless an endpoint and an API key are set.

        For CLIs with no public default endpoint and no local login.
        ``RSI_AGENT_EXTRA_ENV`` may set the CLI's own variables instead.
        """
        extra = self._config.agent_extra_env
        missing = [
            setting
            for setting, value in (
                (
                    "RSI_AGENT_API_BASE_URL",
                    self._config.agent_api_base_url
                    or (self.api_base_env and extra.get(self.api_base_env)),
                ),
                (
                    "RSI_AGENT_API_KEY",
                    self._config.agent_api_key or extra.get(self.api_key_env),
                ),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Agent {self.name!r} uses an OpenAI-compatible endpoint with an "
                f"API key and has no default endpoint; set {' and '.join(missing)}"
            )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        """Return *command* with *effort* applied.

        Only called with a value from ``reasoning_efforts``. Raise
        :class:`ValueError` when the command cannot take it.
        """
        raise ValueError(f"{self.name} does not support reasoning effort")

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        """Install the stop hook into the container.

        Called when the stop hook is enabled.  The default implementation
        logs that no hook is configured (appropriate for agents without one).
        """
        logger.info(
            "No stop hook configured for agent=%s", self.name,
        )
