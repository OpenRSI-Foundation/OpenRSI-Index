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

"""OpenCode, driven over ACP by ``rsi-agent-driver``.

OpenCode has no stop hook at all: its prompt loop ends silently when the
model stops calling tools. The driver sends the continuation as a new prompt
into the same ``opencode acp`` session instead. A Judge's one-shot call runs
OpenCode's own print mode (``opencode run``).
"""

from __future__ import annotations

import json
import shlex

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD, OneShotCommand
from rsi_loop.harness.agent.driven import DrivenAgent, in_scratch_dir

# The platform package, which holds only the native binary; the opencode-ai
# package would pick it in an install script. npm checks the tarball's
# integrity against the registry.
_PACKAGE = f"opencode-linux-x64@{versions.OPENCODE}"
# OpenCode's grep tool downloads ripgrep on first use when no ``rg`` is on
# PATH, which a no-network Work cannot do; the image carries it instead.
_RIPGREP = f"ripgrep-{versions.RIPGREP}-x86_64-unknown-linux-musl"
_RIPGREP_URL = (
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    f"{versions.RIPGREP}/{_RIPGREP}.tar.gz"
)
_RIPGREP_DIR = "/opt/rsi-harness/opencode"
RIPGREP_PATH = f"{_RIPGREP_DIR}/rg"
_PROVIDER = "rsi"
# OpenCode's own config directory, honouring XDG_CONFIG_HOME as it does.
_CONFIG_DIR = '"${XDG_CONFIG_HOME:-$HOME/.config}/opencode"'
# Where OpenCode looks for ripgrep after PATH, so a task's own rg still wins.
_BIN_DIR = '"${XDG_CACHE_HOME:-$HOME/.cache}/opencode/bin"'
# OpenCode installs its plugin SDK into every writable config directory at
# startup and waits for it; without a registry that stalls each start for
# more than a minute. A lock file already naming the package makes it skip
# the install, as root too (a read-only directory would not stop root).
_SKIP_PLUGIN_INSTALL = json.dumps(
    {"packages": {"": {"dependencies": {"@opencode-ai/plugin": "*"}}}}
)
_WEB_TOOLS = ("webfetch", "websearch")
_MODEL_TIMEOUT_MS = 900_000
# What a one-shot call must not take from its working directory. The project
# config switch drops opencode.json, .opencode (agents, commands, plugins,
# skills) and the AGENTS.md, CLAUDE.md and CONTEXT.md walked up from it; the
# skills one the .claude and .agents skill folders, which OpenCode walks up
# to regardless; the Claude Code one CLAUDE.md beside a file the model reads.
_ONESHOT_ENV = (
    "OPENCODE_DISABLE_PROJECT_CONFIG=1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS=1",
    "OPENCODE_DISABLE_CLAUDE_CODE=1",
)
# ``run --format json`` prints one event per line, errors included; each
# finished text part of the session is a ``text`` event with the part last.
# The final message is the last of them, printed as the part object itself,
# whose ``text`` is the answer. Errors go to stderr, where rsi-agent's log
# keeps them (a ``w`` file name runs to the end of its script).
_ERROR_EVENTS = '/^{"type":"error",/w /dev/stderr'
_LAST_TEXT_PART = (
    '/^{"type":"text",/h;'
    '${x;s/^{"type":"text","timestamp":[0-9]*,"sessionID":"[^"]*","part":'
    r"\(.*\)}$/\1/p;}"
)


class OpenCodeAgent(DrivenAgent):
    name = "opencode"
    display_name = "OpenCode"
    protocol = "acp"
    agent_version = versions.OPENCODE
    cli_argv = ("opencode", "acp")
    extra_launchers = ("opencode",)
    # OpenCode compacts by the declared window and sends the declared output
    # limit; it knows neither for a model it does not ship.
    requires_context_limits = True
    # Its default models need an account login or a provider of their own.
    requires_model = True
    install_cmds = [
        NODE_INSTALL_CMD,
        # npm pack fetches the package without running anything from it and
        # honours npm_config_registry; a private cache keeps the tarball out
        # of the image.
        in_scratch_dir(
            f'npm pack --silent --cache "$tmp/cache" {_PACKAGE} >/dev/null',
            f"tar -xzf opencode-linux-x64-{versions.OPENCODE}.tgz package/bin/opencode",
            "install -m 0755 package/bin/opencode /usr/local/bin/opencode",
        ),
        # Only rg is extracted from the release archive, straight into place.
        "sudo -E bash -c "
        + shlex.quote(
            "set -euo pipefail"
            f" && mkdir -p {_RIPGREP_DIR}"
            f" && curl -fsSL {_RIPGREP_URL}"
            " | tar -xzf - --no-same-owner --strip-components=1"
            f" -C {_RIPGREP_DIR} {_RIPGREP}/rg"
            f" && chmod 0755 {RIPGREP_PATH}"
        ),
    ]
    # OpenCode has no variables of its own for a configured provider; its
    # config refers to these by name, so neither the key nor the endpoint is
    # on disk, and the endpoint a no-network Work pins is the one it calls.
    # They are not OPENAI_*, which would also enable OpenCode's own provider.
    api_key_env = "RSI_AGENT_API_KEY"
    api_base_env = "RSI_AGENT_API_BASE_URL"
    # Sent as the OpenAI ``reasoning_effort`` by the OpenAI-compatible
    # provider.
    reasoning_efforts = frozenset({"low", "medium", "high"})

    def check_config(self) -> None:
        super().check_config()
        self.require_api_endpoint()

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Keep a no-network Work from reaching anything but the model
        # endpoint: no update check, models.dev catalog, share links or
        # language-server downloads.
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
        env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
        env["OPENCODE_DISABLE_SHARE"] = "1"
        env["OPENCODE_DISABLE_LSP_DOWNLOAD"] = "1"

    def config(self, *, model: str, internet: bool) -> dict[str, object]:
        """Return the non-secret OpenCode config for a run."""
        window, output = self.context_limits()
        # Every tool runs without asking; nobody answers a question.
        permission = {
            "*": "allow",
            "doom_loop": "allow",
            "external_directory": "allow",
            "question": "deny",
            **{tool: "deny" for tool in (() if internet else _WEB_TOOLS)},
        }
        return {
            "autoupdate": False,
            "share": "disabled",
            # In a git workspace OpenCode would copy the whole worktree into
            # a second repository under $HOME at every step, only to undo
            # edits from its UI.
            "snapshot": False,
            "model": f"{_PROVIDER}/{model}",
            "small_model": f"{_PROVIDER}/{model}",
            "enabled_providers": [_PROVIDER],
            "permission": permission,
            # A title request after the first prompt is extra model traffic.
            "agent": {"title": {"disable": True}},
            "provider": {
                _PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": _PROVIDER,
                    "options": {
                        "baseURL": f"{{env:{self.api_base_env}}}",
                        "apiKey": f"{{env:{self.api_key_env}}}",
                        # A slow reasoning model can stay silent for longer
                        # than the 5 minute defaults; a request aborted
                        # there is retried and would fail the same way.
                        "headerTimeout": _MODEL_TIMEOUT_MS,
                        "chunkTimeout": _MODEL_TIMEOUT_MS,
                    },
                    "models": {
                        model: {
                            "name": model,
                            "tool_call": True,
                            "limit": {"context": window, "output": output},
                            # apply_reasoning_effort fills this in.
                            "options": {},
                        }
                    },
                }
            },
        }

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        if not model:
            raise ValueError(
                f"Agent {self.name!r} needs a model; set RSI_AGENT_MODEL or --model"
            )
        content = json.dumps(self.config(model=model, internet=internet))
        # OPENCODE_CONFIG_CONTENT is merged last, so an opencode.json the task
        # ships cannot change the model, provider or permissions.
        return " && ".join(
            [
                f"mkdir -p {_CONFIG_DIR}/node_modules {_BIN_DIR}",
                f"ln -sf {RIPGREP_PATH} {_BIN_DIR}/rg",
                f"printf '%s' {shlex.quote(_SKIP_PLUGIN_INSTALL)}"
                f" > {_CONFIG_DIR}/package-lock.json",
                f"export OPENCODE_CONFIG_CONTENT={shlex.quote(content)}",
            ]
        )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        marker = '"options": {}'
        if command.count(marker) != 1:
            raise ValueError("OpenCode command cannot accept reasoning effort")
        return command.replace(
            marker, f'"options": {{"reasoningEffort": {json.dumps(effort)}}}', 1
        )

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Work's config (provider, model, limits, permissions, and the marker
        # apply_reasoning_effort rewrites) in the fresh HOME. ``run`` reads
        # the prompt from stdin, auto-rejects any permission request, denies
        # questions and plan mode, and exits once the session is idle.
        prelude = self.config_prelude(
            model=self.effective_model(model), internet=internet
        )
        return OneShotCommand(
            command=(
                f"{prelude} &&\n"
                # OpenCode creates $TMPDIR/opencode at startup, and a Judge's
                # /tmp need not be writable by the call's user.
                'export TMPDIR="$HOME/.tmp" && mkdir -p "$TMPDIR" &&\n'
                f"export {' '.join(_ONESHOT_ENV)} && set -o pipefail &&\n"
                # pipefail keeps OpenCode's status: 1 when the session failed,
                # 127 (no CLI in this Judge) when it is missing.
                "opencode run --format json {schema_option}"
                f" | sed -n -e {shlex.quote(_ERROR_EVENTS)}"
                f" -e {shlex.quote(_LAST_TEXT_PART)}"
            ),
            # ``run`` has no structured-output option: rsi-agent states the
            # schema in the prompt.
            output="stdout-json",
            output_fields=("text",),
        )
