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

"""Kimi Code, driven over ACP by ``rsi-agent-driver``.

Its native Stop hook continues a turn at most once, so the driver sends the
continuation as a new prompt into the same ``kimi acp`` session instead. A
Judge's one-shot call runs Kimi's own print mode (``kimi -p``).
"""

from __future__ import annotations

import json
import shlex

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import NODE_BIN, NODE_INSTALL_CMD, OneShotCommand
from rsi_loop.harness.agent.driven import STDIN_PROMPT_JS, DrivenAgent

_PROVIDER = "rsi"
# The marker apply_reasoning_effort rewrites; see model_toml.
_CAPABILITIES = 'capabilities = ["tool_use"]'
# Tools a headless run must not reach: plan mode makes the session read-only,
# nobody answers questions, and cron jobs and goals start turns of their own
# between the driver's prompts. TaskList, TaskOutput and TaskStop stay: Bash
# runs in the background only while they are enabled.
_DISABLED_TOOLS = (
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "CronCreate",
    "CronDelete",
    "CronList",
    "CreateGoal",
    "UpdateGoal",
    "SetGoalBudget",
    "GetGoal",
)
_WEB_TOOLS = ("FetchURL", "WebSearch")
# Kimi reports a turn that started while another was running as a finished
# prompt and runs it later; the driver then waits for this file to go idle.
_WIRE_GLOB = "$KIMI_CODE_HOME/sessions/*/{session}/agents/main/wire.jsonl"
# A one-shot call's main profile. As an explicit --agent-file it outranks a
# same-name profile in the working directory's .kimi-code/agents or
# .agents/agents, which would otherwise replace Kimi's system prompt;
# ${base_prompt} renders the built-in default one, so the call keeps Work's
# system prompt. No subagents: the working directory can replace their
# profiles too, and no file can outrank those.
_ONESHOT_PROFILE = """---
name: agent
description: Default agent
disallowedTools: Agent, AgentSwarm
---
${base_prompt}
"""
# stream-json prints one line per message and meta lines after the turn; the
# final message is the last assistant line.
_LAST_ASSISTANT_LINE = '/^{"role":"assistant"/h;${x;p;}'


def _toml_string(value: str) -> str:
    # A JSON string is a valid TOML basic string, as long as characters
    # outside the BMP stay literal instead of becoming surrogate escapes.
    return json.dumps(value, ensure_ascii=False)


class KimiCodeAgent(DrivenAgent):
    name = "kimi-code"
    display_name = "Kimi Code"
    protocol = "acp"
    agent_version = versions.KIMI_CODE
    cli_argv = ("kimi", "acp")
    extra_launchers = ("kimi",)
    driver_flags = (
        "--acp-mode",
        "yolo",
        "--idle-probe",
        f"kimi-wire:{_WIRE_GLOB}",
    )
    # Kimi compacts by the declared window; one above the endpoint's makes
    # every turn fail instead.
    requires_context_limits = True
    # Its only default model is the vendor's own, behind an account login.
    requires_model = True
    install_cmds = [
        NODE_INSTALL_CMD,
        "sudo -E npm install -g --ignore-scripts "
        f"@moonshot-ai/kimi-code@{versions.KIMI_CODE}",
    ]
    # Kimi's OpenAI provider falls back to these variables when its config
    # names no base URL, so the config holds neither the endpoint nor the key
    # and the endpoint a no-network Work pins is the one Kimi calls.
    api_key_env = "OPENAI_API_KEY"
    api_base_env = "OPENAI_BASE_URL"
    # Sent as the OpenAI ``reasoning_effort``, without a client-side mapping.
    reasoning_efforts = frozenset({"low", "medium", "high"})

    def check_config(self) -> None:
        super().check_config()
        self.require_api_endpoint()

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Keep a no-network Work from reaching anything but the model
        # endpoint, and keep provider outages inside the turn: Kimi retries
        # every failed model request in place instead of failing the turn.
        env["KIMI_DISABLE_TELEMETRY"] = "1"
        env["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
        env["KIMI_CODE_MODEL_CATALOG_REFRESH_ON_START"] = "0"
        env["KIMI_CODE_MODEL_CATALOG_REFRESH_INTERVAL_MS"] = "0"
        env["KIMI_DISABLE_CRON"] = "1"
        env["KIMI_CODE_INFINITE_RETRY"] = "1"

    def config_toml(self, *, model: str, internet: bool) -> str:
        """Return the non-secret ``config.toml`` for a run."""
        window, output = self.context_limits()
        disabled = [*_DISABLED_TOOLS, *(() if internet else _WEB_TOOLS)]
        return "\n".join(
            [
                f"default_model = {_toml_string(_PROVIDER)}",
                'default_permission_mode = "yolo"',
                "telemetry = false",
                # A title request after every turn is extra model traffic.
                "auto_session_title = false",
                "",
                f"[providers.{_PROVIDER}]",
                'type = "openai"',
                f"api_key_env = {_toml_string(self.api_key_env)}",
                "",
                f"[models.{_PROVIDER}]",
                f"provider = {_toml_string(_PROVIDER)}",
                f"model = {_toml_string(model)}",
                f"max_context_size = {window}",
                f"max_output_size = {output}",
                _CAPABILITIES,
                "",
                "[tools]",
                f"disabled = {json.dumps(disabled)}",
                "",
            ]
        )

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        if not model:
            raise ValueError(
                f"Agent {self.name!r} needs a model; set RSI_AGENT_MODEL or --model"
            )
        toml = self.config_toml(model=model, internet=internet)
        # Exported so the CLI and the driver's idle probe agree on the home.
        return (
            'export KIMI_CODE_HOME="$HOME/.kimi-code" && '
            'mkdir -p "$KIMI_CODE_HOME" && '
            f"printf '%s' {shlex.quote(toml)} > \"$KIMI_CODE_HOME/config.toml\""
        )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        # The model must declare thinking for Kimi to send an effort at all.
        if command.count(_CAPABILITIES) != 1:
            raise ValueError("Kimi Code command cannot accept reasoning effort")
        thinking = "\n".join(
            [
                'capabilities = ["tool_use", "thinking"]',
                f"support_efforts = [{_toml_string(effort)}]",
                f"default_effort = {_toml_string(effort)}",
            ]
        )
        return command.replace(_CAPABILITIES, thinking, 1)

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Work's config.toml (model, limits, telemetry, disabled tools, and the
        # marker apply_reasoning_effort rewrites) in the fresh HOME. Print mode
        # never asks for approval. The explicit profile keeps the working
        # directory's agent profiles out, and --skills-dir its skills; project
        # MCP servers stay off in a folder nobody trusted. Kimi has no switch
        # for the working directory's AGENTS.md, which it always reads.
        prelude = self.config_prelude(
            model=self.effective_model(model), internet=internet
        )
        profile = '"$KIMI_CODE_HOME/oneshot-agent.md"'
        argv = " ".join(
            [
                f"{NODE_BIN} -e",
                shlex.quote(STDIN_PROMPT_JS),
                '-- "$kimi" --output-format stream-json',
                f"--agent-file {profile}",
                '--skills-dir "$KIMI_CODE_HOME/skills"',
                "{schema_option}",
            ]
        )
        return OneShotCommand(
            command=(
                f"{prelude} &&\n"
                f"printf '%s' {shlex.quote(_ONESHOT_PROFILE)} > {profile} &&\n"
                # Work retries a failed model request forever to keep an
                # outage inside its turn; a call fails and the Judge decides.
                "unset KIMI_CODE_INFINITE_RETRY && set -o pipefail &&\n"
                # 127, which rsi-agent reports as no CLI in this Judge.
                "{ kimi=$(command -v kimi) || exit 127; } &&\n"
                f"{argv} | sed -n {shlex.quote(_LAST_ASSISTANT_LINE)}"
            ),
            # No structured-output option: rsi-agent states the schema in the
            # prompt.
            output="stdout-json",
            output_fields=("content",),
        )
