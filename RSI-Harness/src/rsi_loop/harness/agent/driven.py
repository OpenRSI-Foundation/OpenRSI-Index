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

"""Agents kept working by ``rsi-agent-driver`` instead of a native stop hook.

Some CLIs cap or lack a stop hook. For those, the Work image carries one
generic driver (``driver/rsi-agent-driver.mjs``) that starts the CLI, speaks
its ACP or ZCode Protocol, and sends the continuation into the same session
at every turn end. A subclass only describes its CLI.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from rsi_loop.harness.agent.base import NODE_INSTALL_CMD, Agent
from rsi_loop.harness.backend import ContainerBackend, ContainerHandle

DRIVER_PATH = "/opt/rsi-harness/agent-driver.mjs"
DRIVER_LAUNCHER = "/usr/local/bin/rsi-agent-driver"
# For a one-shot call to a print mode that takes the prompt only as an
# argument, where any process can read it and one argument holds at most
# 128 KiB: run as ``node -e STDIN_PROMPT_JS -- MAIN FLAGS...``, it loads the
# CLI's MAIN module into the same node process with the prompt, read from
# stdin, as ``--prompt=`` in its argv instead.
STDIN_PROMPT_JS = (
    "const [main, ...flags] = process.argv.slice(1);"
    ' const prompt = require("fs").readFileSync(0, "utf8");'
    ' process.argv = [process.argv[0], main, ...flags, "--prompt=" + prompt];'
    ' import(require("url").pathToFileURL(main).href);'
)


def in_scratch_dir(*steps: str) -> str:
    """Return a root install command running *steps* in a removed temp dir."""
    script = " && ".join(
        ["set -euo pipefail", "tmp=$(mktemp -d)", 'cd "$tmp"', *steps]
        + ["cd /", 'rm -rf "$tmp"']
    )
    return f"sudo -E bash -c {shlex.quote(script)}"


class DrivenAgent(Agent):
    """Base class for Agents run under ``rsi-agent-driver``.

    Subclasses set ``name``, ``protocol``, ``cli_argv``, ``api_key_env`` and
    extend ``install_cmds``; they may add ``driver_flags`` (or override
    :meth:`driver_args`) and override :meth:`config_prelude` to write the
    CLI's non-secret configuration. When
    ``cli_argv[0]`` is the CLI's own launcher (not a shared interpreter such
    as ``node``), list it in ``extra_launchers`` so a task image's copy cannot
    shadow the pinned one. A Judge's one-shot call (:meth:`format_oneshot_cmd`)
    answers once and exits, so it uses the CLI's own print mode, not the driver.
    """

    # -- required (no default) --
    protocol: str  # "acp" or "zcode"
    cli_argv: tuple[str, ...]  # the CLI command the driver starts

    # -- optional --
    driver_flags: tuple[str, ...] = ()
    agent_version: str | None = None

    install_cmds = [NODE_INSTALL_CMD]
    # Only the launcher matters here: images quarantine and check it, and
    # format_run_cmd renders the real command.
    run_cmd = "rsi-agent-driver --prompt-file {prompt_file}"
    image_files = {
        DRIVER_PATH: ("harness/agent/driver/rsi-agent-driver.mjs", 0o644),
        DRIVER_LAUNCHER: ("harness/agent/driver/rsi-agent-driver.sh", 0o755),
    }

    def config_prelude(self, *, model: str | None, internet: bool) -> str:
        """Return shell commands run before the driver starts, or ``""``.

        Use it to write the CLI's non-secret configuration under ``$HOME``,
        as one ``&&`` chain: the driver starts only when it succeeds.
        Refer to secrets only by environment variable name: Judges and
        snapshots can see the Agent's home directory.
        """
        return ""

    def agent_argv(self, *, model: str | None, internet: bool) -> list[str]:
        """Return the CLI command the driver starts."""
        return list(self.cli_argv)

    def driver_args(self, *, internet: bool) -> list[str]:
        """Return the driver options for this CLI; ``driver_flags`` by default."""
        return list(self.driver_flags)

    def effective_model(self, model: str | None) -> str | None:
        """Return the model a launch names: *model*, the configured or default one.

        Raises :class:`ValueError` when the Agent needs a model and none is set.
        """
        effective = model or self._config.agent_model or self.default_model
        if self.requires_model and not effective:
            raise ValueError(
                f"Agent {self.name!r} needs a model; set RSI_AGENT_MODEL or --model"
            )
        return effective

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
        # A driven CLI keeps one session for the whole run, so there is no
        # resume command: a new run starts a new session.
        effective_model = self.effective_model(model)
        argv = [
            "rsi-agent-driver",
            "--protocol",
            self.protocol,
            *self.driver_args(internet=internet),
            "--agent-name",
            self.name,
        ]
        if self.agent_version:
            argv += ["--agent-version", self.agent_version]
        if effective_model:
            argv += ["--model", effective_model]
        # The prompt is read from the file: task text in argv could be matched
        # by an Agent's own `pkill -f`, and argv is visible to every process.
        argv += ["--prompt-file", prompt_path]
        if continuation is not None:
            argv += ["--continuation-reason", continuation]
        argv += ["--", *self.agent_argv(model=effective_model, internet=internet)]
        command = "exec " + shlex.join(argv)
        prelude = self.config_prelude(model=effective_model, internet=internet).strip()
        # Without its configuration the CLI would start with its own defaults
        # (provider, model, permissions, telemetry), so a failed prelude must
        # not reach the exec. Each prelude is one && chain.
        return f"{prelude} &&\n{command}" if prelude else command

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        # The continuation is a driver flag, so there is no hook file for the
        # Agent to delete; only check the baked-in driver runs as the Agent.
        result = backend.exec_run(
            handle,
            [
                "/bin/sh",
                "-c",
                f"test -x {DRIVER_LAUNCHER} && {DRIVER_LAUNCHER} --version",
            ],
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"rsi-agent-driver is not runnable by the Agent: {result.output}"
            )
        logger.info("Checked rsi-agent-driver for agent=%s", self.name)
