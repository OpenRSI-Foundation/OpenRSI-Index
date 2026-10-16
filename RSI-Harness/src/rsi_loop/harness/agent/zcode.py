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

"""Z.ai ZCode, driven over the ZCode Protocol by ``rsi-agent-driver``.

Its native Stop hook continues a turn at most three times, so the driver
sends the continuation as a new prompt into the same ``zcode app-server``
session instead. Every prompt is a new turn, which also resets ZCode's
per-turn recovery budgets.

ZCode reads an OpenAI-compatible provider's key only from its provider
config file; for that API type it has no environment fallback and no way to
refer to a variable. The prelude therefore writes that one file, from the
environment, into a private directory on the ``/dev/shm`` tmpfs: it stays in
memory, a Judge's rootfs snapshot does not include it, and the rest of the
configuration under ``$HOME`` holds no secret.

A Judge's one-shot call runs ZCode's own print mode (``zcode --prompt``).
rsi-agent gives each call a fresh private HOME on its own tmpfs and deletes
it when the call ends, so the one-shot writes the provider config there: a
directory under ``/dev/shm`` would outlive a killed call, where the Judge and
the next call on the same user could read it.
"""

from __future__ import annotations

import json
import shlex

from rsi_loop.harness.agent import versions
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD, OneShotCommand, select_arch
from rsi_loop.harness.agent.driven import STDIN_PROMPT_JS, DrivenAgent, in_scratch_dir

ZCODE_DIR = "/opt/zcode"
ZCODE_LAUNCHER = "/usr/local/bin/zcode"
# What the launcher runs: the bundle under the node the install pins.
_ZCODE_BUNDLE = f"{ZCODE_DIR}/zcode.cjs"
_ZCODE_NODE = "/usr/local/bin/node"
# The official Linux package for the image's architecture (linux-x64 or
# linux-arm64). Only its headless Node bundle and the built-in provider
# catalog the bundle expects next to it are installed; the desktop app is
# not. A mirror must serve the same <version>/linux-<x64|arm64>/<file> layout.
_DEB = f"ZCode-{versions.ZCODE}-linux-${{arch}}.deb"
_DEB_URL = (
    "${RSI_ZCODE_MIRROR_URL:-https://cdn-zcode.z.ai/zcode/electron/releases}"
    f"/{versions.ZCODE}/linux-${{arch}}/{_DEB}"
)
_RESOURCES = "./opt/ZCode/resources"
_PROVIDER = "rsi"
# The level a run without a reasoning effort selects. ZCode requires one;
# its map adds nothing to the request, so the endpoint's default applies.
_NO_EFFORT = "default"
# Where the provider config with the key lives: a tmpfs, never the disk.
_SECRET_ROOT = "/dev/shm"
# Tools a headless run must not reach: nobody answers questions, plan mode
# waits for a user to leave it, and cron jobs need the desktop host. The
# driver denies the other desktop-only tools itself.
_DENIED_TOOLS = (
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "CronList",
    "CronUpdate",
)
_WEB_TOOLS = ("WebFetch",)
# The desktop-host tools the driver denies in Work.
_DRIVER_DENIED_TOOLS = ("CronCreate", "CronDelete", "OffPeakCreate")
# A one-shot call answers by itself: no skills and no subagents, whose
# listings would show the model the working directory's .zcode/skills,
# .agents/skills and .zcode/agents, and the plugins its config enables.
_ONESHOT_DENIED_TOOLS = ("Skill", "Agent", "SendMessage")
# Print mode reads the working directory's zcode.json and .zcode/config.json
# over the user config, and loads a .env walked up from it, which may set any
# variable Work leaves unset; ZCode has no switch for either. Variables rank
# above the project config, and a .env never replaces one that is set, so the
# one-shot sets those through which the working directory could change the
# call: send model requests, with the key, through a proxy or past a CA of its
# choosing, retry them until rsi-agent's timeout, or move ZCode's files out of
# the private HOME. These keep Work's values; unset, which ZCode reads as
# empty, means no proxy, the default CAs and retries, and no beta storage.
_ONESHOT_PINNED_ENV = (
    "ZCODE_HTTP_PROXY",
    "ZCODE_NO_PROXY",
    "ZCODE_AGENT_CA_CERT",
    "NODE_TLS_REJECT_UNAUTHORIZED",
    "ZCODE_MODEL_RETRY_MAX_RETRIES",
    "ZCODE_MODEL_RETRY_BASE_DELAY_MS",
    "ZCODE_MODEL_RETRY_MAX_DELAY_MS",
    "ZCODE_MODEL_RETRY_BACKOFF_FACTOR",
    "ZCODE_BETA",
    "ZCODE_ENV",
)
# These take ZCode's defaults under the private HOME: some of them read an
# empty value as the working directory.
_ONESHOT_HOME_ENV = {
    "ZCODE_STORAGE_DIR": "$HOME/.zcode",
    "ZCODE_HOME": "$HOME/.zcode",
    "ZCODE_DATA_BASE_DIR": "$HOME",
    "ZCODE_LOG_DIR": "$HOME/.zcode/cli/log",
}
_MODEL_STREAM_IDLE_TIMEOUT_MS = 600_000
# ZCode's own service origin, for everything that is not the model endpoint.
# Port 9 (discard) on loopback refuses at once.
_DEAD_ORIGIN = "http://127.0.0.1:9"
# Fills in the endpoint and key from the environment and writes the provider
# config readable only by the Agent. The key never appears in argv.
_WRITE_PROVIDER_CONFIG = """\
const fs = require('node:fs');
const config = JSON.parse(process.argv[1]);
const key = process.env[process.argv[2]];
const baseUrl = process.env[process.argv[3]];
if (!key || !baseUrl) {
  console.error(`zcode: ${process.argv[2]} and ${process.argv[3]} must be set`);
  process.exit(1);
}
const provider = config.config.providerConfigRules.providerRules[0].config;
provider.access.apiKey = key;
provider.api.baseUrl = baseUrl;
fs.writeFileSync(process.env.ZCODE_PERSONAL_PROVIDER_CONFIG_FILE,
  JSON.stringify(config), { mode: 0o600, flag: 'wx' });
"""


def _reasoning(effort: str | None) -> tuple[str, str]:
    """Return the model rule's level spec and the selection, as JSON members."""
    if effort is None:
        spec = {"values": [_NO_EFFORT], "map": "{}"}
    else:
        # Sent as the OpenAI ``reasoning_effort``, without a client mapping.
        spec = {"values": [effort], "map": '{"reasoning_effort": reasoningLevel}'}
    selection = {"reasoningLevel": effort or _NO_EFFORT}
    return (
        json.dumps({"reasoningLevel": spec})[1:-1],
        json.dumps({"options": selection})[1:-1],
    )


class ZCodeAgent(DrivenAgent):
    name = "zcode"
    display_name = "ZCode"
    protocol = "zcode"
    agent_version = versions.ZCODE
    cli_argv = ("zcode", "app-server")
    extra_launchers = ("zcode",)
    # ZCode compacts by the declared window and sends the declared output
    # limit; it knows neither for a model it does not ship.
    requires_context_limits = True
    # Its default models need a Z.ai account or key.
    requires_model = True
    image_files = {
        **DrivenAgent.image_files,
        ZCODE_LAUNCHER: ("harness/agent/zcode/launcher.sh", 0o755),
    }
    install_cmds = [
        NODE_INSTALL_CMD,
        in_scratch_dir(
            select_arch("zcode", x86_64="x64", aarch64="arm64"),
            # Debian-based images have dpkg-deb; others need ar for the .deb.
            "if ! command -v dpkg-deb >/dev/null && ! command -v ar >/dev/null; "
            "then if command -v apk >/dev/null; then apk add --no-cache binutils; "
            "else apt-get update && DEBIAN_FRONTEND=noninteractive "
            "apt-get install -y binutils && rm -rf /var/lib/apt/lists/*; fi; fi",
            f'curl -fsSL -o zcode.deb "{_DEB_URL}"',
            "{ if command -v dpkg-deb >/dev/null; "
            "then dpkg-deb --fsys-tarfile zcode.deb; "
            "else ar p zcode.deb data.tar.xz | xz -dc; fi; } | tar -xf - "
            f"{_RESOURCES}/glm {_RESOURCES}/config/provider/zcode-builtin.json",
            "rm zcode.deb",
            f"mkdir -p {ZCODE_DIR} && cp -R {_RESOURCES}/glm/. {ZCODE_DIR}/",
            f"install -D -m 0644 {_RESOURCES}/config/provider/zcode-builtin.json "
            f"{ZCODE_DIR}/provider/zcode-builtin.json",
            f"chmod -R a+rX,go-w {ZCODE_DIR}",
        ),
    ]
    # The prelude copies these into the provider config, which cannot refer
    # to a variable; the endpoint a no-network Work pins is the one used.
    api_key_env = "RSI_AGENT_API_KEY"
    api_base_env = "RSI_AGENT_API_BASE_URL"
    reasoning_efforts = frozenset({"low", "medium", "high"})

    def check_config(self) -> None:
        super().check_config()
        self.require_api_endpoint()

    def build_args(self) -> dict[str, str | None]:
        return {"RSI_ZCODE_MIRROR_URL": self._config.zcode_mirror_url or None}

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        # Model telemetry would otherwise go to an OTLP endpoint when one is
        # configured in the environment.
        env["ZCODE_MODEL_TELEMETRY_ENABLED"] = "false"
        # Before each request to an https endpoint, ZCode fetches a routing
        # table from its vendor origin, which may redirect the request, with
        # its key, elsewhere. A dead loopback origin makes that fetch fail
        # locally, so requests go unrouted and nothing else leaves Work.
        env["ZCODE_BASE_URL"] = _DEAD_ORIGIN

    def driver_args(self, *, internet: bool) -> list[str]:
        denied = [*_DENIED_TOOLS, *(() if internet else _WEB_TOOLS)]
        return ["--zcode-tool-deny", ",".join(denied)]

    def cli_config(self) -> dict[str, object]:
        """Return the non-secret ``~/.zcode/cli/config.json`` for a run."""
        return {
            "permission": {"mode": "yolo"},
            # Memory consolidation sends extra model requests.
            "memory": {"use": False},
            # The bundled plugins drive a desktop, browser or device that a
            # headless Work does not have.
            "plugins": {"enabled": False},
            "modelStream": {"idleTimeoutMs": _MODEL_STREAM_IDLE_TIMEOUT_MS},
        }

    def provider_config(self, *, model: str) -> dict[str, object]:
        """Return the provider config without the endpoint and key.

        The prelude fills those in from the environment when it writes the
        file.
        """
        window, output = self.context_limits()
        return {
            "schemaVersion": 1,
            "config": {
                "providerOrder": [_PROVIDER],
                "providerConfigRules": {
                    "providerRules": [
                        {
                            "providerId": _PROVIDER,
                            "templateId": None,
                            "providerName": _PROVIDER,
                            "enabled": True,
                            "config": {
                                "group": "standard-personal",
                                "logo": None,
                                "access": {"type": "api-key", "apiKey": None},
                                "api": {
                                    "type": "openai-chat-completions",
                                    "baseUrl": None,
                                    "headers": {},
                                },
                                "personalModelIds": [model],
                                "modelOrder": [model],
                                "visibility": "visible",
                            },
                        }
                    ]
                },
                "modelConfigRules": {
                    "providerModelRules": [],
                    "manualProviderModelRules": [
                        {
                            "providerId": _PROVIDER,
                            "modelId": model,
                            "config": {
                                "enabled": True,
                                "properties": {
                                    "contextWindow": window,
                                    "inputFormat": {
                                        "supportsImage": False,
                                        "supportsVideo": False,
                                        "supportsPdf": False,
                                    },
                                    "supportsJsonSchemaOutput": False,
                                    "supportsNativeWebSearch": False,
                                    "supportsMidConversationSystem": False,
                                },
                                "optionSpecs": {
                                    "maxOutputTokens": {"max": output},
                                    "reasoningLevel": {
                                        "values": [_NO_EFFORT],
                                        "map": "{}",
                                    },
                                },
                            },
                        }
                    ],
                },
                "defaultModelSelection": {
                    "providerId": _PROVIDER,
                    "modelId": model,
                    "options": {"reasoningLevel": _NO_EFFORT},
                },
            },
        }

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        return self._prelude(
            model,
            tmpfs=_SECRET_ROOT,
            secret_dir=f"$(mktemp -d {_SECRET_ROOT}/rsi-zcode.XXXXXX)",
        )

    def _prelude(self, model: str | None, *, tmpfs: str, secret_dir: str) -> str:
        """Write the CLI config, and the provider config into *secret_dir*.

        *secret_dir* must lie on the tmpfs mounted at *tmpfs*.
        """
        if not model:
            raise ValueError(
                f"Agent {self.name!r} needs a model; set RSI_AGENT_MODEL or --model"
            )
        cli_config = json.dumps(self.cli_config())
        provider = json.dumps(self.provider_config(model=model))
        return " && ".join(
            [
                'mkdir -p "$HOME/.zcode/cli"',
                f"printf '%s\\n' {shlex.quote(cli_config)}"
                ' > "$HOME/.zcode/cli/config.json"',
                # Refuse to write the key anywhere but a tmpfs.
                f'test "$(stat -f -c %T {tmpfs})" = tmpfs',
                f"rsi_secret_dir={secret_dir}",
                "export ZCODE_PERSONAL_PROVIDER_CONFIG_FILE="
                '"$rsi_secret_dir/provider_config.json"',
                "node -e "
                + shlex.join(
                    [
                        _WRITE_PROVIDER_CONFIG,
                        provider,
                        self.api_key_env,
                        str(self.api_base_env),
                    ]
                ),
            ]
        )

    def apply_reasoning_effort(self, command: str, effort: str) -> str:
        markers = _reasoning(None)
        if any(command.count(marker) != 1 for marker in markers):
            raise ValueError("ZCode command cannot accept reasoning effort")
        for marker, replacement in zip(markers, _reasoning(effort), strict=True):
            command = command.replace(marker, replacement, 1)
        return command

    def format_oneshot_cmd(
        self,
        *,
        model: str | None = None,
        internet: bool = True,
    ) -> OneShotCommand:
        # Work's CLI and provider config (model, limits, memory and plugins
        # off, and the markers apply_reasoning_effort rewrites), with the
        # provider config in the fresh HOME. Print mode answers one prompt in
        # one turn, in yolo mode, with no title request. ZCode has no switch
        # for the working directory's AGENTS.md, which it always reads, nor
        # for its project config, which can still turn on plugins and start
        # MCP servers; it cannot change the model, endpoint, key or limits.
        prelude = self._prelude(
            self.effective_model(model), tmpfs='"$HOME"', secret_dir='"$HOME/.zcode"'
        )
        denied = [
            *_DENIED_TOOLS,
            *_DRIVER_DENIED_TOOLS,
            *_ONESHOT_DENIED_TOOLS,
            *(() if internet else _WEB_TOOLS),
        ]
        pinned = " ".join(
            [
                *(f'{name}="${{{name}-}}"' for name in _ONESHOT_PINNED_ENV),
                *(f'{name}="{value}"' for name, value in _ONESHOT_HOME_ENV.items()),
            ]
        )
        argv = " ".join(
            [
                _ZCODE_NODE,
                "-e",
                shlex.quote(STDIN_PROMPT_JS),
                f"-- {_ZCODE_BUNDLE} --output-format json --mode yolo",
                f"--disallowed-tools {','.join(denied)}",
                "{schema_option}",
            ]
        )
        return OneShotCommand(
            command=(
                f"{prelude} &&\n"
                f"export {pinned} &&\n"
                # 127, which rsi-agent reports as no CLI in this Judge.
                f"{{ test -f {_ZCODE_BUNDLE} || exit 127; }} &&\n"
                f"{argv}"
            ),
            # One JSON object whose ``response`` is the final message; a
            # failed call exits 1 with the error on stderr. No structured
            # output option: rsi-agent states the schema in the prompt.
            output="stdout-json",
            output_fields=("response",),
        )
