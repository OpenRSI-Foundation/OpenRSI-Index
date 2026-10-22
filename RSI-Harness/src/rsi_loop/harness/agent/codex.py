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

"""Codex agent — internet blocking and stop hook."""

from __future__ import annotations

import json
import logging
import re
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


_PROVIDER_FLAGS = (
    "-c model_provider=rsi_proxy "
    "-c model_providers.rsi_proxy.name=rsi_proxy "
    """-c 'model_providers.rsi_proxy.base_url="'"$OPENAI_BASE_URL"'"' """
    "-c model_providers.rsi_proxy.env_key=OPENAI_API_KEY "
    "-c model_providers.rsi_proxy.wire_api=responses"
)
# Rollouts are written under ~/.codex/sessions; ``codex exec resume`` finds a
# session by its rollout alone.
_SESSIONS = (".codex/sessions", ".codex-sessions")
_SESSION_STORE = link_session_store(*_SESSIONS)
# The header line ``codex exec`` prints for every session it starts or resumes.
_SESSION_LINE = re.compile(r"session id: ([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})")
# FORCE_COLOR in Work's environment makes it bold even without a TTY.
_SGR = re.compile(r"\x1b\[[0-9;]*m")


class CodexAgent(Agent):

    name = "codex"
    install_cmds = [
        NODE_INSTALL_CMD,
        f"sudo -E npm install -g @openai/codex@{versions.CODEX}",
        '''if [ -n "$OPENAI_BASE_URL" ]; then
    mkdir -p ~/.codex
    cat > ~/.codex/config.toml << EOF
model_provider = "rsi-loop-proxy"
model_verbosity = "medium"
model = "${CODEX_MODEL:-gpt-5.5}"

[model_providers.rsi-loop-proxy]
name = "rsi-loop-proxy"
base_url = "${OPENAI_BASE_URL}"
env_key = "OPENAI_API_KEY"
EOF
fi''',
    ]
    run_cmd = 'codex exec --dangerously-bypass-approvals-and-sandbox "$(cat {prompt_file})"'
    resume_cmd = 'codex exec resume --last --dangerously-bypass-approvals-and-sandbox "Continue working."'
    resumes_sessions = True
    session_store = _SESSIONS
    api_key_env = "OPENAI_API_KEY"
    api_base_env = "OPENAI_BASE_URL"
    default_api_base_url = "https://api.openai.com"
    model_env = "CODEX_MODEL"
    stop_hook = "codex"
    display_name = "Codex"
    reasoning_efforts = frozenset(
        {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    )

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        if not self._config.agent_api_base_url:
            if "OPENAI_API_KEY" in env and "CODEX_API_KEY" not in env:
                env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]

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
            # The next user turn is read from stdin ("-").
            cmd = (
                "codex exec resume --dangerously-bypass-approvals-and-sandbox"
                f" {shlex.quote(session_id)} - <{prompt_path}"
            )
        else:
            cmd = super().format_run_cmd(
                prompt_path, model=model, cwd=cwd, internet=internet, resume=resume,
            )
        cmd = self._launch_flags(cmd, model=model, internet=internet)
        return f"{_SESSION_STORE} &&\nexec {cmd}"

    def session_id_in(self, line: str) -> str | None:
        match = _SESSION_LINE.fullmatch(_SGR.sub("", line.rstrip("\r\n")))
        return match.group(1) if match else None

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # No session, rules, hooks, project docs or user config: files in the
        # working directory cannot change how the call behaves. The prompt
        # arrives on stdin ("-") and the final message goes to a file. Goal
        # tools need a persistent thread, so an ephemeral call never offers
        # them (a model kept calling them until its timeout).
        cmd = (
            "codex exec --ephemeral --skip-git-repo-check --ignore-user-config"
            " --ignore-rules --dangerously-bypass-approvals-and-sandbox"
            " -c features.hooks=false -c features.goals=false"
            " -c project_doc_max_bytes=0"
            " --output-last-message {output_file} {schema_option} -"
        )
        return OneShotCommand(
            command=self._launch_flags(cmd, model=model, internet=internet),
            schema_option="--output-schema {schema_file}",
            output="file",
        )

    def _launch_flags(
        self, cmd: str, *, model: str | None, internet: bool
    ) -> str:
        """Model, provider and web-search flags shared by every launch."""
        marker = _flags_marker(cmd)
        if model:
            cmd = cmd.replace(marker, f"{marker} --model {shlex.quote(model)}", 1)

        if self._config.agent_api_base_url or self._config.agent_extra_env.get(
            "OPENAI_BASE_URL"
        ):
            # The install-time config.toml never sees OPENAI_BASE_URL (images
            # build without the Agent environment), so name the provider at
            # launch. The URL and key stay in the environment, not argv.
            cmd = cmd.replace(marker, f"{marker} {_PROVIDER_FLAGS}", 1)

        # Codex otherwise saves its shell's environment, API key included,
        # under $HOME, where a Judge's snapshot of Work would capture it.
        cmd = cmd.replace(marker, f"{marker} -c features.shell_snapshot=false", 1)

        # Codex runs the model's commands in a login shell by default, whose
        # profile may reset PATH (Debian's does): the commands would lose the
        # image's ENV PATH and the sandbox client directory the Agent has.
        cmd = cmd.replace(marker, f"{marker} -c allow_login_shell=false", 1)

        if not internet:
            cmd = cmd.replace(marker, f'{marker} -c web_search="disabled"', 1)

        return cmd

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        if "codex exec" not in command:
            raise ValueError("RSI Loop Codex command cannot accept reasoning effort")
        marker = _flags_marker(command)
        override = shlex.quote(f'model_reasoning_effort="{effort}"')
        return command.replace(marker, f"{marker} -c {override}", 1)

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        hook_path = "/tmp/rsi-loop-codex-stop-hook.sh"

        hook_script = _generate_codex_stop_hook()
        local_hook = log_dir / "_codex-stop-hook.sh"
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
                f"Failed to set Codex stop hook permissions: {result.output}"
            )
        result = backend.exec_run(
            handle,
            ["/bin/sh", "-c", f"test -r {hook_path} && test -x {hook_path}"],
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Codex stop hook is not accessible to the Agent: {result.output}"
            )
        logger.info("Installed Codex stop hook")

        hooks_content = _generate_codex_hooks(hook_path)
        local_hooks = log_dir / "_codex_hooks.json"
        local_hooks.write_text(hooks_content)
        backend.exec_run(handle, "mkdir -p /etc/codex", user="root")
        backend.copy_to_container(
            handle,
            local_hooks,
            PurePosixPath("/etc/codex/hooks.json"),
        )

        config_append = _generate_codex_config_append()
        local_config_append = log_dir / "_codex_config_append.toml"
        local_config_append.write_text(config_append)
        backend.copy_to_container(
            handle,
            local_config_append,
            PurePosixPath("/tmp/rsi-loop-codex-config-append.toml"),
        )

        enable_hooks_cmd = r"""
set -e
CONFIG=/home/agent/.codex/config.toml
TMP=$(mktemp)
mkdir -p /home/agent/.codex
touch "$CONFIG"
if grep -Eq '^[[:space:]]*hooks[[:space:]]*=' "$CONFIG"; then
    sed -E 's/^[[:space:]]*hooks[[:space:]]*=.*/hooks = true/' "$CONFIG" > "$TMP"
elif grep -Eq '^[[:space:]]*\[features\][[:space:]]*$' "$CONFIG"; then
    awk '
        /^[[:space:]]*\[features\][[:space:]]*$/ {
            print
            print "hooks = true"
            next
        }
        { print }
    ' "$CONFIG" > "$TMP"
else
    cp "$CONFIG" "$TMP"
    cat /tmp/rsi-loop-codex-config-append.toml >> "$TMP"
fi
mv "$TMP" "$CONFIG"
chown -R agent:agent /home/agent/.codex
"""
        result = backend.exec_run(
            handle, ["/bin/bash", "-lc", enable_hooks_cmd], user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to configure Codex hooks: {result.output}")
        logger.info("Configured Codex hooks")


def _flags_marker(command: str) -> str:
    """Where launch flags go: ``codex exec resume`` takes them after ``resume``."""
    return "codex exec resume" if "codex exec resume" in command else "codex exec"


# ---------------------------------------------------------------------------
# Script generators (private to this module)
# ---------------------------------------------------------------------------


def _generate_codex_stop_hook() -> str:
    output = json.dumps(
        {"decision": "block", "reason": STOP_HOOK_REASON}, separators=(",", ":")
    )
    return f"""#!/bin/bash
cat >/dev/null
echo {shlex.quote(output)}
"""


def _generate_codex_hooks(hook_path: str) -> str:
    hooks: dict = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_path,
                            "timeout": 30,
                            "statusMessage": "RSI Loop stop hook",
                        }
                    ]
                }
            ],
        }
    }
    return json.dumps(hooks, indent=2)


def _generate_codex_config_append() -> str:
    return """\n[features]\nhooks = true\n"""
