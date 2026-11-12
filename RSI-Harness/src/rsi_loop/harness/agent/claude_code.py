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

"""Claude Code agent — env augmentation, model flag, and stop hook."""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path, PurePosixPath

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import (
    NODE_INSTALL_CMD,
    STOP_HOOK_REASON,
    Agent,
    OneShotCommand,
    link_session_store,
    remove_managed_config,
)
from rsi_loop.harness.backend import ContainerBackend, ContainerHandle

# Transcripts are written under ~/.claude/projects; ``--resume`` finds a
# session by its transcript alone.
_SESSIONS = (".claude/projects", ".claude-projects")
_SESSION_STORE = link_session_store(*_SESSIONS)
_WEB_TOOLS = ("WebSearch", "WebFetch")
# Tools that come with a claude.ai login (some with an API key too) and act
# outside Work: cloud routines, claude.ai projects, designs and artifacts,
# notifications, the account's MCP connectors and registry, files or messages
# to other sessions, the user's browser; and tools that start turns or work
# outside this process: scheduled jobs, workflows, worktrees. Denied tools
# stay out of subagents too. Without SendMessage a finished subagent cannot
# be continued. (SendToProject and SendToProjectContext of older releases are
# Projects in the pinned one.) The claude.ai plugin and skill catalogs,
# connector suggestions, team onboarding links, linked inboxes and feedback
# reach the account as well. Names a release does not have are ignored.
_ACCOUNT_TOOLS = (
    "RemoteTrigger",
    "Projects",
    "ClaudeDesign",
    "DesignSync",
    "PushNotification",
    "Artifact",
    "ArtifactComments",
    "ArtifactData",
    "ArtifactCheck",
    "AppifactRepl",
    "ReadNotifications",
    "ListConnectors",
    "SearchMcpRegistry",
    "SendFile",
    "SendUserFile",
    "OfferChromeSetup",
    "ListAgents",
    "SendMessage",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "Workflow",
    "EnterWorktree",
    "ExitWorktree",
    "ShareOnboardingGuide",
    "SearchPlugins",
    "SearchSkills",
    "ListPlugins",
    "ListSkills",
    "SuggestPluginInstall",
    "SuggestSkills",
    "PublishPlugin",
    "SuggestConnectors",
    "FetchInboxMessage",
    "SendFeedback",
)
# The same switches for what is not a tool: claude.ai MCP connectors, org
# memory, the Remote Control bridge, workflows and artifacts.
_ACCOUNT_SETTINGS = {
    "disableClaudeAiConnectors": True,
    "disableRemoteControl": True,
    "remoteControlAtStartup": False,
    "disableWorkflows": True,
    "disableArtifact": True,
}
_ACCOUNT_ENV = {
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "CLAUDE_CODE_DISABLE_ORG_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_ARTIFACT": "1",
    "CLAUDE_CODE_DISABLE_CRON": "1",
    "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
}

# The model, its default and the aliases (opus, sonnet, haiku, fable, the
# small fast model) a settings file's ``env`` could point at another model.
# Pinned to the operator's value from the run's environment, or to "", which
# Claude Code reads as unset: its own default and alias resolution.
_MODEL_ENV = (
    "ANTHROPIC_DEFAULT_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
)
# Managed settings, MCP servers and memory outrank --settings.
_MANAGED_CONFIG = remove_managed_config("/etc/claude-code")


def _same_model_env(model: str | None) -> dict[str, str]:
    """Every subagent runs on the main model.

    FORCE drops the model an Agent tool call, an agent definition's
    ``model:`` or a workflow names, and removes the tool's ``model``
    parameter; subagents then take CLAUDE_CODE_SUBAGENT_MODEL, the run's
    model (``inherit``, the main model, when the run names none). Explore's
    own cap and the advisor tool, which consults another model, are off, and
    so is the main model's substitution when it hits a usage limit.
    """
    return {
        "CLAUDE_CODE_SUBAGENT_MODEL": model or "inherit",
        "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
        "CLAUDE_CODE_DISABLE_EXPLORE_INHERIT_CAP": "1",
        "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
        "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    }


class ClaudeCodeAgent(Agent):

    name = "claude-code"
    install_cmds = [
        NODE_INSTALL_CMD,
        f"sudo -E npm install -g @anthropic-ai/claude-code@{versions.CLAUDE_CODE}",
    ]
    # The prompt is fed on stdin, not as an argument: with the task text in
    # argv, an Agent running `pkill -f <name mentioned in the task>` matches
    # and kills its own claude process (observed: `pkill -f run_fixed.py`).
    run_cmd = (
        "claude -p"
        " --output-format stream-json"
        " --verbose"
        " --dangerously-skip-permissions"
        " <{prompt_file}"
    )
    resume_cmd = (
        'claude --continue -p "Continue working."'
        " --output-format stream-json"
        " --verbose"
        " --dangerously-skip-permissions"
    )
    resumes_sessions = True
    session_store = _SESSIONS
    api_key_env = "ANTHROPIC_AUTH_TOKEN"
    api_base_env = "ANTHROPIC_BASE_URL"
    default_api_base_url = "https://api.anthropic.com"
    model_env = "ANTHROPIC_MODEL"
    stop_hook = "claude"
    display_name = "Claude Code"
    reasoning_efforts = frozenset({"low", "medium", "high", "xhigh", "max"})

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # The Work container is the sandbox. Claude Code refuses
        # --dangerously-skip-permissions as root unless IS_SANDBOX is exactly "1",
        # and task images without a USER run the Agent as root.
        env["IS_SANDBOX"] = "1"
        # Harness owns the wall-clock budget. Do not let Claude's consecutive
        # Stop-hook block cap end the Agent early while that budget remains.
        env["CLAUDE_CODE_STOP_HOOK_BLOCK_CAP"] = "0"
        if self._config.claude_cache_opt:
            env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        if not self._config.agent_api_base_url and self._config.agent_api_key:
            env["ANTHROPIC_API_KEY"] = self._config.agent_api_key
        # After RSI_AGENT_EXTRA_ENV, which cannot undo them. The launch's
        # --settings repeats them, over any settings file the Agent writes.
        env.update(self._guard_env(model))

    def _effective_model(self, model: str | None) -> str | None:
        return model or self._config.agent_model or self.default_model

    def _main_model(self, model: str | None) -> str:
        """The run's model, the operator's ANTHROPIC_MODEL, or "" (the CLI's
        default for the login)."""
        return self._effective_model(model) or self._config.agent_extra_env.get(
            "ANTHROPIC_MODEL", ""
        )

    def _guard_env(self, model: str | None) -> dict[str, str]:
        extra = self._config.agent_extra_env
        return {
            "ANTHROPIC_MODEL": self._main_model(model),
            **{key: str(extra.get(key, "")) for key in _MODEL_ENV},
            **_same_model_env(self._effective_model(model)),
            **_ACCOUNT_ENV,
        }

    def _guard_flags(
        self, model: str | None, *, internet: bool, work: bool = False
    ) -> str:
        """The flags every launch takes: no account-level tools, MCP servers
        only from this command line (none), and one --settings.

        Flag settings rank above user, project and local settings, whose
        ``env`` and ``model`` would otherwise override the process
        environment: an Agent could write ``.claude/settings.json`` turning
        the subagent pin off, remapping an alias or naming another model for
        a run without ``--model``. Only managed settings rank higher, and the
        launch removes them.
        """
        denied = [*_ACCOUNT_TOOLS, *(() if internet else _WEB_TOOLS)]
        settings: dict[str, object] = {
            "model": self._main_model(model),
            "env": self._guard_env(model),
            **_ACCOUNT_SETTINGS,
        }
        if work and self._config.claude_cache_opt:
            settings["includeGitInstructions"] = False
        text = json.dumps(settings, separators=(",", ":"))
        return (
            f" --strict-mcp-config --disallowedTools {','.join(denied)}"
            f" --settings {shlex.quote(text)}"
        )

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
        # The stop hook file installed by install_stop_hook carries the
        # continuation, so the command does not.
        if resume and session_id is not None:
            cmd = self.run_cmd.format(prompt_file=prompt_path, cwd=cwd).replace(
                "claude ", f"claude --resume {shlex.quote(session_id)} ", 1
            )
        else:
            cmd = super().format_run_cmd(
                prompt_path, model=model, cwd=cwd, internet=internet, resume=resume,
            )
            if not resume and session_id is not None:
                cmd += f" --session-id {shlex.quote(session_id)}"

        effective_model = self._effective_model(model)
        if effective_model:
            cmd += f" --model {shlex.quote(effective_model)}"

        if self._config.claude_cache_opt:
            cmd += " --exclude-dynamic-system-prompt-sections"

        cmd += self._guard_flags(effective_model, internet=internet, work=True)
        return f"{_SESSION_STORE} && {_MANAGED_CONFIG} &&\nexec {cmd}"

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Print mode with one JSON result on stdout. Safe mode and user-only
        # settings keep CLAUDE.md, hooks, plugins and project settings in the
        # working directory from changing the call; no session is kept. The
        # prompt arrives on stdin.
        cmd = (
            "claude -p --output-format json --no-session-persistence"
            " --dangerously-skip-permissions --safe-mode --setting-sources user"
            " {schema_option}"
        )
        effective_model = self._effective_model(model)
        if effective_model:
            cmd += f" --model {shlex.quote(effective_model)}"
        cmd += self._guard_flags(effective_model, internet=internet)
        return OneShotCommand(
            command=f"{_MANAGED_CONFIG} &&\nexec {cmd}",
            schema_option='--json-schema "$(cat {schema_file})"',
            output="stdout-json",
            output_fields=("structured_output", "result"),
            error_field="is_error",
        )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        # The launch is the last line, after the session store and managed
        # config steps; a bare command is the launch alone.
        prelude, newline, launch = command.rpartition("\n")
        marker = "exec claude " if newline else "claude "
        if not launch.startswith(marker):
            raise ValueError(
                "RSI Loop Claude Code command cannot accept reasoning effort"
            )
        return (
            f"{prelude}{newline}{marker}--effort {shlex.quote(effort)} "
            f"{launch[len(marker) :]}"
        )

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        hook_path = "/tmp/rsi-loop-stop-hook.sh"

        hook_script = _generate_stop_hook()
        local_hook = log_dir / "_stop-hook.sh"
        local_hook.write_text(hook_script)
        # Apptainer binds this source read-only, so set its mode before copying.
        local_hook.chmod(0o755)
        backend.copy_to_container(handle, local_hook, PurePosixPath(hook_path))
        # Docker copies reset the mode to 0644; executable read-only binds need
        # no mutation. Check both setup and access as the actual Agent user.
        result = backend.exec_run(
            handle,
            ["/bin/sh", "-c", f"test -x {hook_path} || chmod a+rx {hook_path}"],
            user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to set Claude Code stop hook permissions: {result.output}"
            )
        result = backend.exec_run(
            handle,
            ["/bin/sh", "-c", f"test -r {hook_path} && test -x {hook_path}"],
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Claude Code stop hook is not accessible to the Agent: {result.output}"
            )
        logger.info("Installed Claude Code stop hook")

        settings_content = _generate_claude_settings(hook_path)
        local_settings = log_dir / "_claude_settings.json"
        local_settings.write_text(settings_content)
        backend.copy_to_container(
            handle,
            local_settings,
            PurePosixPath("/home/agent/.claude/settings.json"),
        )
        logger.info("Configured Claude Code settings with hooks")


# ---------------------------------------------------------------------------
# Script generators (private to this module)
# ---------------------------------------------------------------------------


def _generate_stop_hook() -> str:
    output = json.dumps(
        {"decision": "block", "reason": STOP_HOOK_REASON}, separators=(",", ":")
    )
    return f"""#!/bin/bash
cat >/dev/null
echo {shlex.quote(output)}
"""


def _generate_claude_settings(hook_path: str) -> str:
    settings: dict = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_path,
                        }
                    ]
                }
            ],
        }
    }
    return json.dumps(settings, indent=2)
