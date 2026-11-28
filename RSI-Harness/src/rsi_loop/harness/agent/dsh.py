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

"""DeepSeek Harness (``dsh``), driven over ACP by ``rsi-agent-driver``.

Work runs its ``acp`` profile; a Judge's one-shot call runs its ``headless``
profile, which reads the prompt on stdin and prints the final answer. dsh
composes each profile from plugin patch layers, and the last layer before
the command line is ``$DSH_HOME/cordis.patch.yml``: every launch rewrites it
with the run's model route, a fresh profile directory beneath it.
"""

from __future__ import annotations

import json
import shlex

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD, OneShotCommand
from rsi_loop.harness.agent.driven import DrivenAgent

_PROVIDER = "rsi"
# Written as JSON, which YAML reads, except for this placeholder: a ``!!js``
# expression dsh evaluates at launch, so the endpoint stays off disk and the
# one a no-network Work pins is the one dsh calls.
_BASE_URL = "@RSI_BASE_URL@"
# The model's reasoning switch, last in the route, which
# apply_reasoning_effort rewrites together with the route's default level.
_NO_EFFORT = '"reasoningEfforts": false}]'
# Plugins a run must not mount, by profile entry id: the vendor's own
# account, routes, search and log upload; the title request after a prompt,
# extra model traffic; goals, which start turns of their own between the
# driver's prompts; and live config reload, so a patch the Agent edits loads
# only at the next launch, which rewrites it.
_DISABLED = (
    "llm-deepseek",
    "llm-deepseek-account",
    "deepseek-account",
    "credentials",
    "authorization",
    "web-search-deepseek",
    "web-deepseek-api",
    "session-log-deepseek",
    "session-title-llm",
    "goal",
    "goal-round-driver",
    "command-goal",
    "tool-goal",
    "hmr",
)
_WEB_TOOLS = ("tool-web",)
# A one-shot call answers from the prompt alone: no workspace instructions
# or skills, which the working directory could supply, and no subagents.
_ONESHOT_DISABLED = (
    "agent-instructions",
    "skill",
    "skill-filesystem",
    "tool-skill",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-subagent-control",
    "tool-workflow",
    "tool-agent-team",
)
# Files under $DSH_HOME a launch would load besides its own patch: the
# profile (its plugins and patch layer) and settings an earlier release
# left, imported into a new profile.
_STALE = ("profiles", "settings.yaml")
# The headless profile's --json stream: the last final event.
_FINAL_LINE = '/^{"type":"final"/h;${x;p;}'
# The shell tool's BASH_ENV, which restores the variables dsh drops from
# every command's environment by name (see dsh/shell-env.sh).
SHELL_ENV_PATH = "/opt/rsi-harness/dsh-shell-env.sh"
_KEEP_SHELL_ENV = (
    "RSI_DSH_SHELL_ENV= && "
    'for name in $(compgen -e); do case "${name^^}" in DSH_*) ;; '
    "*KEY*|*PASSWORD*|*SECRET*|*TOKEN*) "
    'RSI_DSH_SHELL_ENV+="$(declare -p "$name")"$\'\\n\' ;; esac; done && '
    f'if [ -r {SHELL_ENV_PATH} ] && [ "${{BASH_ENV-}}" != {SHELL_ENV_PATH} ]; '
    'then export RSI_DSH_SHELL_ENV RSI_DSH_BASH_ENV="${BASH_ENV-}" '
    f"BASH_ENV={SHELL_ENV_PATH}; fi"
)


class DshAgent(DrivenAgent):
    name = "dsh"
    display_name = "DeepSeek Harness"
    protocol = "acp"
    agent_version = versions.DSH
    cli_argv = ("dsh", "--profile", "acp")
    extra_launchers = ("dsh",)
    # dsh compacts by the declared window and sends the declared output
    # limit with every request.
    requires_context_limits = True
    # Its default models are the vendor's own, behind a key or login.
    requires_model = True
    image_files = {
        **DrivenAgent.image_files,
        SHELL_ENV_PATH: ("harness/agent/dsh/shell-env.sh", 0o644),
    }
    install_cmds = [
        NODE_INSTALL_CMD,
        f"sudo -E npm install -g --ignore-scripts @deepseek-ai/dsh@{versions.DSH}",
    ]
    # Not OPENAI_*, which other routes read as their own key.
    api_key_env = "RSI_AGENT_API_KEY"
    api_base_env = "RSI_AGENT_API_BASE_URL"
    # Sent as the OpenAI ``reasoning_effort``, without a client-side mapping.
    reasoning_efforts = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})

    def check_config(self) -> None:
        super().check_config()
        self.require_api_endpoint()

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Its permission preset: no sandbox of its own (Work is one) and no
        # approval prompts.
        env["DSH_PERMISSION_MODE"] = "danger-full-access"
        env["DSH_TELEMETRY_DISABLED"] = "1"
        # Its native addons load from the install, not from copies it would
        # otherwise make under the temp directory, which may be noexec.
        env["NARB_DISABLE_NATIVE_CACHE"] = "1"

    def patch_yaml(self, *, model: str, internet: bool, oneshot: bool = False) -> str:
        """Return the non-secret ``cordis.patch.yml`` for a launch."""
        window, output = self.context_limits()
        route: dict[str, object] = {
            "apiKeyEnv": self.api_key_env,
            "api": "openai-completions",
            "baseURL": _BASE_URL,
            # The plain chat-completions request every compatible endpoint
            # accepts: a system message instead of OpenAI's developer role,
            # and no stored-completion flag.
            "compat": {"supportsDeveloperRole": False, "supportsStore": False},
        }
        if not oneshot:
            # Keep provider outages inside the turn: retry every failed model
            # request in place. A call fails after a few, and the Judge
            # decides.
            route["retryPolicy"] = {"mode": "always"}
        route["models"] = [
            {
                "id": model,
                "contextWindow": window,
                "maxTokens": output,
                "reasoningEfforts": False,
            }
        ]
        selection = {"provider": _PROVIDER, "model": model}
        rows: list[dict[str, object]] = [
            {"id": "llm-pi-ai", "config": {"providers": {_PROVIDER: route}}},
            # Subagents, workflows and team members start on this selection,
            # and the route holds the run's model alone.
            {"id": "agent-default-model", "config": selection},
        ]
        if not oneshot:
            rows.append({"id": "acp", "config": selection})
        disabled = [
            *_DISABLED,
            *(() if internet else _WEB_TOOLS),
            *(_ONESHOT_DISABLED if oneshot else ()),
        ]
        rows += [{"id": entry, "disabled": True} for entry in disabled]
        text = "".join(f"- {json.dumps(row, ensure_ascii=False)}\n" for row in rows)
        expression = json.dumps(f"process.env.{self.api_base_env}")
        return text.replace(json.dumps(_BASE_URL), f"!!js {expression}", 1)

    def _prelude(self, *, model: str | None, internet: bool, oneshot: bool) -> str:
        if not model:
            raise ValueError(
                f"Agent {self.name!r} needs a model; set RSI_AGENT_MODEL or --model"
            )
        patch = self.patch_yaml(model=model, internet=internet, oneshot=oneshot)
        stale = " ".join(f'"$DSH_HOME/{name}"' for name in _STALE)
        return (
            'export DSH_HOME="$HOME/.dsh" && '
            f"rm -rf {stale} && "
            'mkdir -p "$DSH_HOME" && '
            f"printf '%s' {shlex.quote(patch)} > \"$DSH_HOME/cordis.patch.yml\" && "
            f"{_KEEP_SHELL_ENV}"
        )

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        return self._prelude(model=model, internet=internet, oneshot=False)

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        # The model must declare the level, and the route's default level is
        # the one a session uses.
        if command.count(_NO_EFFORT) != 1:
            raise ValueError("dsh command cannot accept reasoning effort")
        level = json.dumps(effort)
        return command.replace(
            _NO_EFFORT,
            f'"reasoningEfforts": {{{level}: {level}}}}}], "reasoning": {level}',
            1,
        )

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Work's route, limits and marker apply_reasoning_effort rewrites, in
        # the fresh HOME. The headless profile never asks for approval under
        # this permission preset and exits 1 when the call failed. Its --json
        # stream ends with one final event holding the answer exactly; plain
        # output adds a newline.
        prelude = self._prelude(
            model=self.effective_model(model), internet=internet, oneshot=True
        )
        return OneShotCommand(
            command=(
                f"{prelude} &&\n"
                "set -o pipefail &&\n"
                # 127, which rsi-agent reports as no CLI in this Judge.
                "{ dsh=$(command -v dsh) || exit 127; } &&\n"
                '"$dsh" --profile headless --json {schema_option}'
                f" | sed -n {shlex.quote(_FINAL_LINE)}"
            ),
            # No structured-output option: rsi-agent states the schema in the
            # prompt.
            output="stdout-json",
            output_fields=("text",),
        )
