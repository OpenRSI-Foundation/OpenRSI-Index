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
)
from rsi_loop.harness.backend import ContainerBackend, ContainerHandle

# Transcripts are written under ~/.claude/projects; ``--resume`` finds a
# session by its transcript alone.
_SESSIONS = (".claude/projects", ".claude-projects")
_SESSION_STORE = link_session_store(*_SESSIONS)


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

        effective_model = (
            model or self._config.agent_model or self.default_model
        )
        if effective_model:
            cmd += f" --model {shlex.quote(effective_model)}"

        if not internet:
            cmd += " --disallowedTools WebSearch,WebFetch"

        if self._config.claude_cache_opt:
            cmd += (
                " --exclude-dynamic-system-prompt-sections"
                """ --settings '{"includeGitInstructions":false}'"""
            )

        return f"{_SESSION_STORE} &&\nexec {cmd}"

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
        effective_model = model or self._config.agent_model or self.default_model
        if effective_model:
            cmd += f" --model {shlex.quote(effective_model)}"
        if not internet:
            cmd += " --disallowedTools WebSearch,WebFetch"
        return OneShotCommand(
            command=cmd,
            schema_option='--json-schema "$(cat {schema_file})"',
            output="stdout-json",
            output_fields=("structured_output", "result"),
            error_field="is_error",
        )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        # The launch is the last line, after the session store step of a Work
        # command; a one-shot command is the launch alone.
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
