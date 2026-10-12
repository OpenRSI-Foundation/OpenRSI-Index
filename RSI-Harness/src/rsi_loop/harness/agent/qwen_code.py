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

"""Qwen Code, driven over ACP by ``rsi-agent-driver``.

Its native Stop hook is capped at 100 blocks per prompt, so the driver sends
the continuation as a new prompt into the same ``qwen --acp`` session instead.
ACP rather than stream-json: in stream-json the first failed model call runs
Qwen's exit cleanup (killing background jobs) and plan mode can park a turn
forever.
"""

from __future__ import annotations

import json
import shlex

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD, OneShotCommand
from rsi_loop.harness.agent.driven import DrivenAgent

# Tools a headless run must not reach: plan mode waits for a user to approve
# leaving it, and nobody answers questions.
_DENIED_TOOLS = ("enter_plan_mode", "exit_plan_mode", "ask_user_question")
_WEB_TOOLS = ("web_fetch", "web_search")
# Safe mode keeps the skill tool, which lists the working directory's
# .qwen/commands by their descriptions and loads a command's body on request:
# a one-shot call would pass instructions from a Judge's evidence to the model.
_ONESHOT_DENIED_TOOLS = ("skill",)
# The tools that ask before they run (reading outside the working directory
# included). A one-shot call cannot use yolo mode (see format_oneshot_cmd),
# so it allows them by name; nobody could confirm them.
_CONFIRMED_TOOLS = (
    "read_file",
    "glob",
    "grep_search",
    "run_shell_command",
    "edit",
    "write_file",
    "notebook_edit",
)


def _denied_tools(*, internet: bool) -> list[str]:
    return [*_DENIED_TOOLS, *(() if internet else _WEB_TOOLS)]


def _settings_prelude(settings: dict[str, object]) -> str:
    text = json.dumps(settings, sort_keys=True)
    return (
        'mkdir -p "$HOME/.qwen" && '
        f"printf '%s\\n' {shlex.quote(text)} > \"$HOME/.qwen/settings.json\""
    )


class QwenCodeAgent(DrivenAgent):
    name = "qwen-code"
    display_name = "Qwen Code"
    protocol = "acp"
    agent_version = versions.QWEN_CODE
    cli_argv = ("qwen", "--acp", "--approval-mode", "yolo")
    extra_launchers = ("qwen",)
    # yolo makes most tools run without asking. The permission requests that
    # remain get the option that returns to yolo first.
    driver_flags = (
        "--permission-prefer",
        "restore_previous,allow_always,allow_once",
    )
    # Qwen assumes a default window for a model it does not know; a larger
    # one than the endpoint's makes it overflow instead of compacting.
    requires_context_limits = True
    # Without one Qwen sends its own vendor model name, which the required
    # custom endpoint does not serve.
    requires_model = True
    install_cmds = [
        NODE_INSTALL_CMD,
        "sudo -E npm install -g --ignore-scripts "
        f"@qwen-code/qwen-code@{versions.QWEN_CODE}",
    ]
    # Qwen reads its OpenAI-compatible provider from these variables. Using
    # them directly means the endpoint a no-network Work pins is the one Qwen
    # calls, and the key exists only in the environment.
    api_key_env = "OPENAI_API_KEY"
    api_base_env = "OPENAI_BASE_URL"
    model_env = "OPENAI_MODEL"

    def check_config(self) -> None:
        super().check_config()
        self.require_api_endpoint()

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Keep a no-network Work from reaching anything but the model
        # endpoint: no models.dev catalog fetch, telemetry or title requests.
        env["QWEN_CODE_MODELS_DEV"] = "off"
        env["QWEN_TELEMETRY_ENABLED"] = "false"
        env["QWEN_DISABLE_AUTO_TITLE"] = "1"
        env["QWEN_CODE_SUPPRESS_YOLO_WARNING"] = "1"

    def settings(self, *, internet: bool) -> dict[str, object]:
        """Return the non-secret ``~/.qwen/settings.json`` for a run."""
        window, output = self.context_limits()
        denied = _denied_tools(internet=internet)
        # Never set maxSessionTurns, maxWallTimeSeconds, maxToolCalls or
        # sessionTokenLimit: they count across every prompt of the session.
        return {
            "security": {"auth": {"selectedType": "openai"}},
            "general": {"enableAutoUpdate": False},
            "privacy": {"usageStatisticsEnabled": False},
            "telemetry": {"enabled": False},
            # Each of these sends extra model requests after every turn.
            "memory": {"enableManagedAutoMemory": False},
            "ui": {"enableFollowupSuggestions": False},
            # A model-created cron job would start turns of its own between
            # the driver's prompts.
            "experimental": {"cron": False},
            "permissions": {"deny": denied},
            "model": {
                "generationConfig": {
                    "contextWindowSize": window,
                    "samplingParams": {"max_tokens": output},
                }
            },
        }

    def oneshot_settings(self, *, internet: bool) -> dict[str, object]:
        """Return the Work settings with folder trust on, for a one-shot call.

        With folder trust on and no folder trusted, Qwen ignores the working
        directory's ``.qwen/settings.json`` and ``.env``, which a Judge's
        evidence may carry; safe mode alone still reads both.
        """
        settings = self.settings(internet=internet)
        security = settings["security"]
        assert isinstance(security, dict)
        security["folderTrust"] = {"enabled": True}
        return settings

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        return _settings_prelude(self.settings(internet=internet))

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Print mode: the prompt on stdin, one stream-json result line out.
        # Safe mode drops QWEN.md, hooks, extensions, skills and MCP servers,
        # and the untrusted folder the working directory's settings; neither
        # keeps a session. An untrusted folder also refuses yolo mode, so the
        # tools that ask are allowed by name. Safe mode ignores the settings'
        # permissions, hence the deny list again as --exclude-tools, plus the
        # skill tool, through which project commands still reach the model.
        argv = [
            "qwen",
            "--safe-mode",
            "--chat-recording=false",
            "--output-format",
            "stream-json",
            "--allowed-tools",
            ",".join([*_CONFIRMED_TOOLS, *(_WEB_TOOLS if internet else ())]),
            "--exclude-tools",
            ",".join([*_denied_tools(internet=internet), *_ONESHOT_DENIED_TOOLS]),
        ]
        effective_model = self.effective_model(model)
        if effective_model:
            argv += ["--model", effective_model]
        prelude = _settings_prelude(self.oneshot_settings(internet=internet))
        # Only the last line, the result: the whole event stream could pass
        # rsi-agent's stdout limit and lose it. pipefail keeps Qwen's status.
        return OneShotCommand(
            command=(
                f"{prelude} && set -o pipefail &&\n"
                f"{shlex.join(argv)} {{schema_option}} | tail -n 1"
            ),
            # Qwen ends the call at the first answer that conforms.
            schema_option="--json-schema @{schema_file}",
            output="stdout-json",
            output_fields=("structured_result", "result"),
            error_field="is_error",
        )
