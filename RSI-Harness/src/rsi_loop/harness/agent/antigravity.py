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

"""Antigravity CLI (``agy``), driven over ACP by ``rsi-agent-driver``.

Its native Stop hook continues a turn at most ten times, so the driver sends
the continuation as a new prompt into the same ``agy acp`` session instead.
Over ACP agy takes only a model of its own catalog, and only through
session/set_model, which the driver sends. A Judge's one-shot call runs
agy's print mode.

It authenticates with a Gemini API key or, with ``--agent-auth local``, the
operator's own Google login: agy's token file, which the Harness puts on a
private tmpfs and the launch links into agy's directory.
"""

from __future__ import annotations

import json
import re
import shlex

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import (
    NODE_BIN,
    NODE_INSTALL_CMD,
    OneShotCommand,
    remove_managed_config,
    select_arch,
)
from rsi_loop.harness.agent.driven import DrivenAgent, in_scratch_dir

# The glibc build for the image's architecture: the pinned node, and so the
# driver, needs glibc anyway. The archive holds the one binary.
_SELECT_RELEASE = select_arch(
    "Antigravity",
    x86_64="linux-x64/cli_linux_x64",
    aarch64="linux-arm/cli_linux_arm64",
)
_RELEASE_URL = (
    "https://storage.googleapis.com/antigravity-public/antigravity-cli/"
    f"{versions.ANTIGRAVITY}-{versions.ANTIGRAVITY_BUILD}/${{arch}}.tar.gz"
)
# Managed config, which can name other models and a model gateway.
_MANAGED_CONFIG = remove_managed_config("/etc/antigravity")
# The other ways agy picks its credentials or endpoint: a Google Cloud login,
# a model gateway, Vertex AI, or another key the Gemini SDK prefers.
_OTHER_ENDPOINTS = (
    "AGY_ADC_AUTH",
    "AGY_LLM_GATEWAY_URL",
    "AGY_LLM_GATEWAY_API_KEY",
    "AGY_LLM_GATEWAY_MODELS",
    "AGY_LLM_GATEWAY_HEADERS",
    "AGY_LLM_GATEWAY_CA_CERT",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
)
_SETTINGS_DIR = '"$HOME/.gemini/antigravity-cli"'
_CONFIG_DIR = '"$HOME/.gemini/config"'
# API-key auth with the run's endpoint. No auto mode or terminal sandbox:
# both need namespaces, and in Work every command would fail.
_SETTINGS = {"modelProvider": "gemini", "agentMode": "", "enableTerminalSandbox": False}
# A Google login: without modelProvider agy uses its token file and ignores
# GEMINI_API_KEY.
_LOGIN_SETTINGS = {"agentMode": "", "enableTerminalSandbox": False}
# The login as the Harness places it under $HOME (a private tmpfs in Work,
# the call's home in a Judge), and where agy reads it. agy rewrites the
# file in place when it refreshes the token, so the refreshed token stays on
# the tmpfs.
LOGIN_PATH = ".gemini/antigravity-login/antigravity-oauth-token"
_LOGIN_LINK = '"$HOME/.gemini/antigravity-cli/antigravity-oauth-token"'
# Under a login agy's logs name the account's email, so they go to the
# login's tmpfs too, off the rootfs a Judge's snapshot copies; cli.log is a
# relative link into this directory.
_LOG_LINK = '"$HOME/.gemini/antigravity-cli/log"'
# What else could pick another credential or endpoint under a login: the
# API key and endpoints, and another account.
_LOGIN_UNSET = (
    "GEMINI_API_KEY",
    "GOOGLE_GEMINI_BASE_URL",
    "CLOUD_CODE_URL",
    "AGY_ACCOUNT",
    "AGY_CDE_SEAMLESS_AUTH",
)
# What the model asks for beyond the main model, answered by a global
# PreToolUse hook (it applies with permissions skipped too): every subagent
# runs as the built-in self (the main agent's setup) on the main model, as
# an agent definition's own model and a built-in type's fixed one outrank
# the call's; and the tools that start agents on any model (swarm,
# run_workflow), call another model (search_web, generate_image,
# browser_subagent), start turns of their own (schedule), message other
# conversations (send_message, manage_inbox) or reach the account's plugins
# (list_plugin_accounts, search_marketplace) are denied.
_DENY = '{"decision":"deny","reason":"disabled in this run"}'
_SELF = '{"decision":"allow","overwrite":{"Model":"inherit","TypeName":"self"}}'
_RULES = (
    ("invoke_subagent", _SELF),
    (
        "swarm|run_workflow|search_web|generate_image|browser_subagent"
        "|schedule|send_message|manage_inbox|list_plugin_accounts"
        "|search_marketplace",
        _DENY,
    ),
)
# Under a login the model host also fetches web pages for read_url_content,
# past a no-network Work's firewall.
_LOGIN_OFFLINE_RULES = (("read_url_content", _DENY),)


def _hooks(rules: tuple[tuple[str, str], ...]) -> dict:
    return {
        "rsi-harness": {
            "PreToolUse": [
                {
                    "matcher": matcher,
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"cat >/dev/null; echo {shlex.quote(reply)}",
                        }
                    ],
                }
                for matcher, reply in rules
            ]
        }
    }


# How agy reports a failed model call over ACP: as the turn's reply.
_ERROR_TEXT = "⚠ agent executor error:"
# Print mode cuts a longer prompt (about 191 KB with its own wrapping)
# without a word, and with it the schema rsi-agent adds at the end.
_MAX_PROMPT_BYTES = 180_000
# Print mode joins every reply of the turn; its stream-json sends each reply
# as the text deltas of one step. This prints the last reply, without the
# newline agy adds, or the turn's error.
_LAST_REPLY_JS = (
    "const texts = {}; let result = null;"
    ' for (const line of require("fs").readFileSync(0, "utf8").split("\\n")) {'
    " let event; try { event = JSON.parse(line); } catch { continue; }"
    " const step = event.step_update;"
    ' if (step?.step_type === "agent_response" && typeof step.text_delta === "string")'
    ' texts[step.step_index] = (texts[step.step_index] ?? "") + step.text_delta;'
    ' if (event.event === "result") result = event.result; }'
    ' const reply = Object.values(texts).filter((text) => text.trim()).pop() ?? "";'
    ' console.log(JSON.stringify(result?.status === "SUCCESS"'
    ' ? { response: reply.replace(/\\n$/, "") }'
    ' : { error: true, response: String(result?.error ?? "no result") }));'
)
# A model flag of the run or one-shot command, which apply_reasoning_effort
# rewrites; and the effort suffix of a catalog id.
_MODEL_FLAG = re.compile(r"(--(?:acp-)?model )([A-Za-z0-9._-]+)")
_EFFORT = re.compile(r"-(?:low|medium|high|xhigh)$")


class AntigravityAgent(DrivenAgent):
    name = "antigravity"
    display_name = "Antigravity"
    protocol = "acp"
    agent_version = versions.ANTIGRAVITY
    # No permission requests; telemetry off (an ACP-only switch).
    cli_argv = ("agy", "acp", "--dangerously-skip-permissions", "--disable-telemetry")
    extra_launchers = ("agy",)
    driver_flags = ("--acp-error-text", _ERROR_TEXT)
    acp_model = True
    # A catalog id such as gemini-4-carbon-high; agy maps it to the model
    # and thinking level it sends. Its catalog fixes the limits too.
    requires_model = True
    install_cmds = [
        NODE_INSTALL_CMD,
        in_scratch_dir(
            _SELECT_RELEASE,
            f'curl -fsSL "{_RELEASE_URL}" | tar -xz antigravity',
            "install -m 0755 antigravity /usr/local/bin/agy",
        ),
    ]
    # agy appends /v1beta/models/... to the base URL and sends the key as
    # X-Goog-Api-Key; the endpoint must speak the Gemini API.
    api_key_env = "GEMINI_API_KEY"
    api_base_env = "GOOGLE_GEMINI_BASE_URL"
    default_api_base_url = "https://generativelanguage.googleapis.com"
    # The suffix of the catalog id, which apply_reasoning_effort rewrites.
    reasoning_efforts = frozenset({"low", "medium", "high", "xhigh"})

    def check_config(self) -> None:
        super().check_config()
        if not (
            self._config.agent_local_auth
            or self._config.agent_api_key
            or self._config.agent_extra_env.get(self.api_key_env)
        ):
            raise ValueError(
                f"Agent {self.name!r} uses a Gemini API key or a local login; "
                "set RSI_AGENT_API_KEY or pass --agent-auth local"
            )
        if self._config.agent_local_auth and any(
            self._config.agent_extra_env.get(name)
            for name in (self.api_key_env, "GOOGLE_API_KEY")
        ):
            # Both would reach Work, and the login would win.
            raise ValueError(
                f"Agent {self.name!r} takes a Gemini key or a local login, not "
                "both; drop the key from RSI_AGENT_EXTRA_ENV or --agent-auth local"
            )

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Its binary directory is root-owned anyway; the check alone would
        # reach the update server.
        env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        # search_web is denied whatever the network: it calls another model.
        login = self._config.agent_local_auth
        unset = _OTHER_ENDPOINTS + (_LOGIN_UNSET if login else ())
        steps = [
            _MANAGED_CONFIG,
            f"unset {' '.join(unset)}",
            f"mkdir -p {_SETTINGS_DIR} {_CONFIG_DIR}",
        ]
        if login:
            # A fresh link each launch: whatever the Agent left in its place
            # (a file on the rootfs, which a Judge's snapshot would carry)
            # goes.
            steps += [
                f'{{ [ -f "$HOME/{LOGIN_PATH}" ] || {{ echo "rsi-harness: the'
                ' Antigravity login is missing" >&2; exit 1; }; }',
                f"rm -rf -- {_LOGIN_LINK} {_LOG_LINK}",
                f"ln -s ../antigravity-login/antigravity-oauth-token {_LOGIN_LINK}",
                'mkdir -p "$HOME/.gemini/antigravity-login/log"',
                f"ln -s ../antigravity-login/log {_LOG_LINK}",
            ]
        settings = _LOGIN_SETTINGS if login else _SETTINGS
        hooks = _hooks(
            _RULES + (_LOGIN_OFFLINE_RULES if login and not internet else ())
        )
        steps += [
            f"printf '%s' {shlex.quote(json.dumps(settings))}"
            f" > {_SETTINGS_DIR}/settings.json",
            f"printf '%s' {shlex.quote(json.dumps(hooks))} > {_CONFIG_DIR}/hooks.json",
            f"printf '%s' {shlex.quote(json.dumps({'mcpServers': {}}))}"
            f" > {_CONFIG_DIR}/mcp_config.json",
        ]
        return " && ".join(steps)

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        # A catalog id that lacks the combination fails at setup.
        if not _MODEL_FLAG.search(command):
            raise ValueError("Antigravity command cannot accept reasoning effort")
        return _MODEL_FLAG.sub(
            lambda match: f"{match[1]}{_EFFORT.sub('', match[2])}-{effort}", command
        )

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Work's settings and hooks in the fresh HOME. With no prompt
        # argument and stdin not a terminal, agy reads the prompt from stdin
        # (print mode), answers and exits, nonzero when the model call
        # failed. Tools run without asking, as in Work. agy always reads
        # the working directory's AGENTS.md, GEMINI.md, skills and
        # .agents/hooks.json; there is no switch for them. Print mode has no
        # telemetry switch: a no-network Judge blocks it.
        effective_model = self.effective_model(model)
        assert effective_model is not None
        argv = [
            "agy",
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--model",
            effective_model,
            "--output-format",
            "stream-json",
        ]
        prelude = self.config_prelude(model=effective_model, internet=internet)
        return OneShotCommand(
            command=(
                f"{prelude} &&\n"
                'prompt="$HOME/prompt.md" && cat >"$prompt" &&\n'
                f'if [ "$(wc -c <"$prompt")" -gt {_MAX_PROMPT_BYTES} ]; then'
                f" echo 'rsi-harness: Antigravity takes a prompt of at most"
                f" {_MAX_PROMPT_BYTES} bytes whole' >&2; exit 2; fi &&\n"
                # pipefail keeps agy's status: 3 when the model call failed,
                # 127 when this Judge has no agy.
                f"set -o pipefail && {shlex.join(argv)} {{schema_option}}"
                f' <"$prompt" | {NODE_BIN} -e {shlex.quote(_LAST_REPLY_JS)}'
            ),
            # Its --json-schema answer comes with every earlier reply of the
            # turn: rsi-agent states the schema in the prompt instead.
            output="stdout-json",
            output_fields=("response",),
            error_field="error",
        )
