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

"""pi, kept working by a pi extension baked into its Work image.

pi has an uncapped native stop point, the extension event
``agent_before_settle``, so it needs no driver: ``pi/rsi-never-stop.ts``
queues the continuation as a follow-up user message at every settle, in the
same process and session, and writes the normalized transcript that the
driver writes for driven Agents. A Judge's one-shot call runs pi's own print
mode without the extension.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD, Agent, OneShotCommand
from rsi_loop.harness.backend import ContainerBackend, ContainerHandle

EXTENSION_PATH = "/opt/rsi-harness/pi/rsi-never-stop.ts"
_PROVIDER = "rsi"
# pi's own default; exported so pi and the prelude agree on it.
_AGENT_DIR = '"$HOME/.pi/agent"'
# The marker apply_reasoning_effort rewrites; see models_config.
_NO_REASONING = '"reasoning": false'
# A slow reasoning model can stay silent for longer than pi's 5 minute
# default; a request aborted there is retried and would fail the same way.
_HTTP_IDLE_TIMEOUT_MS = 600_000
# Fills in the endpoint from the environment: pi interpolates variables only
# in the key and headers, and the key stays a reference to its variable.
_WRITE_MODELS = """\
const fs = require('node:fs');
const config = JSON.parse(process.argv[1]);
const baseUrl = process.env[process.argv[2]];
if (!baseUrl) {
  console.error(`pi: ${process.argv[2]} must be set`);
  process.exit(1);
}
for (const provider of Object.values(config.providers)) provider.baseUrl = baseUrl;
fs.writeFileSync(process.argv[3], JSON.stringify(config, null, 2) + '\\n');
"""
# What a one-shot call must not take from its working directory or load at
# all: no session file, none of the project's trust-gated .pi resources
# (settings, SYSTEM.md, extensions, skills, prompt templates, MCP servers),
# no AGENTS.md or CLAUDE.md from it or its parents, and no extension, skill
# or prompt template from anywhere, so a prompt starting with ``/`` is sent
# as plain text.
_ONESHOT_FLAGS = (
    "--no-session",
    "--no-approve",
    "--no-context-files",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
)


class PiAgent(Agent):
    name = "pi"
    display_name = "pi"
    install_cmds = [
        NODE_INSTALL_CMD,
        "sudo -E npm install -g --ignore-scripts "
        f"@earendil-works/pi-coding-agent@{versions.PI}",
    ]
    # Only the launcher matters here: images quarantine and check it, and
    # format_run_cmd renders the real command.
    run_cmd = "pi --offline -p <{prompt_file}"
    image_files = {EXTENSION_PATH: ("harness/agent/pi/rsi-never-stop.ts", 0o644)}
    # The config refers to the key by this name, so it is never on disk, and
    # the endpoint a no-network Work pins is the one pi calls. They are not
    # OPENAI_*, which would also enable pi's own OpenAI provider.
    api_key_env = "RSI_AGENT_API_KEY"
    api_base_env = "RSI_AGENT_API_BASE_URL"
    # pi compacts by the declared window and sends the declared output limit;
    # with a window above the endpoint's, overflow recovery finds nothing to
    # compact and every request fails.
    requires_context_limits = True
    # Its default models need an account login or a provider key of their own.
    requires_model = True
    # Sent as the OpenAI ``reasoning_effort``, without a client-side mapping.
    reasoning_efforts = frozenset({"minimal", "low", "medium", "high"})

    def check_config(self) -> None:
        super().check_config()
        self.require_api_endpoint()

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Keep a no-network Work from reaching anything but the model
        # endpoint: no catalog refresh, version check or install telemetry.
        env["PI_OFFLINE"] = "1"
        env["PI_SKIP_VERSION_CHECK"] = "1"
        env["PI_TELEMETRY"] = "0"
        # Reported in the transcript's init line.
        env["RSI_AGENT_VERSION"] = versions.PI

    def settings(self) -> dict[str, object]:
        """Return pi's ``settings.json`` for a run."""
        return {
            "compaction": {"enabled": True},
            "httpIdleTimeoutMs": _HTTP_IDLE_TIMEOUT_MS,
            # Retries inside a turn; the extension backs off between turns.
            "retry": {
                "enabled": True,
                "maxRetries": 5,
                "baseDelayMs": 2000,
                "maxAgentDelayMs": 60_000,
            },
            "enableInstallTelemetry": False,
            # The llama.cpp extension probes a local server and the MCP one
            # starts configured servers; a headless run uses neither.
            "extensions": ["-builtin:llama.cpp", "-builtin:mcp"],
        }

    def models_config(self, *, model: str) -> dict[str, object]:
        """Return ``models.json`` without the endpoint, which the prelude adds."""
        window, output = self.context_limits()
        return {
            "providers": {
                _PROVIDER: {
                    "baseUrl": None,
                    "api": "openai-completions",
                    "apiKey": f"${self.api_key_env}",
                    # The plain chat-completions request every compatible
                    # endpoint accepts: a system message instead of OpenAI's
                    # developer role, and no stored-completion flag.
                    "compat": {"supportsDeveloperRole": False, "supportsStore": False},
                    "models": [
                        {
                            "id": model,
                            "name": model,
                            "contextWindow": window,
                            "maxTokens": output,
                            # apply_reasoning_effort turns this on.
                            "reasoning": False,
                        }
                    ],
                }
            }
        }

    def _effective_model(self, model: str | None) -> str:
        effective_model = model or self._config.agent_model or self.default_model
        if not effective_model:
            raise ValueError(
                f"Agent {self.name!r} needs a model; set RSI_AGENT_MODEL or --model"
            )
        return effective_model

    def _prelude(self, model: str) -> list[str]:
        """Write pi's settings and models into the agent directory under HOME."""
        models = json.dumps(self.models_config(model=model))
        settings = json.dumps(self.settings())
        return [
            f"export PI_CODING_AGENT_DIR={_AGENT_DIR}",
            'mkdir -p "$PI_CODING_AGENT_DIR"',
            f"printf '%s\\n' {shlex.quote(settings)}"
            ' > "$PI_CODING_AGENT_DIR/settings.json"',
            "node -e "
            + shlex.join([_WRITE_MODELS, models, str(self.api_base_env)])
            + ' "$PI_CODING_AGENT_DIR/models.json"',
        ]

    def format_run_cmd(
        self,
        prompt_path: str,
        *,
        model: str | None = None,
        cwd: str = "",
        internet: bool = True,
        resume: bool = False,
        continuation: str | None = None,
    ) -> str:
        # pi keeps one session for the whole run, so there is no resume
        # command: a new run starts a new session.
        effective_model = self._effective_model(model)
        prelude = self._prelude(effective_model)
        # The extension continues only while this is set, that is, while the
        # run's stop hook is enabled; it always writes the transcript.
        if continuation is not None:
            prelude.append(
                f"export RSI_CONTINUATION_REASON={shlex.quote(continuation)}"
            )
        argv = [
            "pi",
            "--offline",
            "-p",
            "-e",
            EXTENSION_PATH,
            "--provider",
            _PROVIDER,
            "--model",
            effective_model,
        ]
        # The prompt is read from the file: task text in argv could be matched
        # by an Agent's own `pkill -f`, and argv is visible to every process.
        # A file, not an open pipe, also lets pi's print mode reach its end.
        command = f"exec {shlex.join(argv)} <{shlex.quote(prompt_path)}"
        # Without its configuration pi would start with its own defaults, so a
        # failed prelude must not reach the exec.
        return " && ".join(prelude) + " &&\n" + command

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # pi has no web tools, so a no-network call differs from an online
        # one only in the environment, which pins the endpoint.
        del internet
        effective_model = self._effective_model(model)
        # Work's launch without the extension, which is what keeps Work going:
        # print mode answers the prompt on stdin once and exits, nonzero when
        # the final response failed.
        argv = [
            "pi",
            "--offline",
            "-p",
            *_ONESHOT_FLAGS,
            "--provider",
            _PROVIDER,
            "--model",
            effective_model,
        ]
        return OneShotCommand(
            command=(
                # Work's settings and models (the endpoint, limits and the
                # marker apply_reasoning_effort rewrites) in the fresh HOME.
                " && ".join(self._prelude(effective_model)) + " &&\n"
                # pi writes the full text of a long tool output to the temp
                # directory: keep it private to the call and removed with
                # the HOME, and writable when a Judge's /tmp is not.
                'export TMPDIR="$HOME/.tmp" && mkdir -p "$TMPDIR" &&\n'
                # Print mode ends every text block of the final response with
                # a newline, which the substitution drops; it keeps pi's
                # status (127 when this Judge has no pi), and the exec keeps
                # the launcher apply_reasoning_effort finds.
                f"answer=$(exec {shlex.join(argv)} {{schema_option}}) &&\n"
                'printf "%s" "$answer" >{output_file}'
            ),
            # pi has no structured-output option: rsi-agent states the
            # schema in the prompt.
        )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        launcher = "exec pi "
        if command.count(_NO_REASONING) != 1 or command.count(launcher) != 1:
            raise ValueError("pi command cannot accept reasoning effort")
        command = command.replace(_NO_REASONING, '"reasoning": true', 1)
        return command.replace(
            launcher, f"{launcher}--thinking {shlex.quote(effort)} ", 1
        )

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        # The continuation is set by the command, so there is no hook file for
        # the Agent to delete; only check the baked-in extension and pi itself
        # work for the Agent.
        result = backend.exec_run(
            handle,
            ["/bin/sh", "-c", f"test -r {EXTENSION_PATH} && pi --version"],
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"pi or its extension is not usable by the Agent: {result.output}"
            )
        logger.info("Checked pi and its extension for agent=%s", self.name)
