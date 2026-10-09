"""``rsi-harness`` with a scripted Work agent, for the operator acceptance.

The CLI chooses an Agent adapter (codex, claude-code), and choosing or
writing agents is not Harness work, so there is no scripted agent to name.
This runs the real ``rsi_harness.cli`` app (argument parsing, policy
loading, ProductionRuntimeServices, the real iptables firewall, Work and
Judge containers, the sandbox endpoints) with only the codex adapter's model
CLI replaced by a bash script, as tests/integration/test_sandbox_e2e.py does:
the pinned install commands become a stub launcher, and the Agent command
becomes the script. Everything the script does in Work (Harbor through the
plugin, ``rsi-sandbox``, ``rsi-submit``) goes through the production path.

    python -m tests.acceptance.scripted_cli --work-script FILE \\
        [--work-files DIR ...] [--work-env KEY=VALUE ...] -- \\
        run TASK --agent codex --sandbox-policy POLICY [rsi-harness options]

Each ``--work-files DIR`` goes into one tar (as ``<DIR name>/``), copied
into Work as ``$RSI_ACCEPTANCE_FILES`` before the script starts (the
sample's /tests exists only in the Judge); ``--work-env`` adds variables to
the script's environment. Arguments after ``--`` are ``rsi-harness``'s own.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

FILES = PurePosixPath("/tmp/rsi-acceptance-files.tar")
# The launcher the Work image build checks for; never a model CLI.
STUB_LAUNCHER = (
    "printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/codex && "
    "chmod 0755 /usr/local/bin/codex"
)


def bundle(directories: list[Path], archive: Path) -> None:
    """One tar of the directories, each as ``<name>/``."""
    with tarfile.open(archive, "w") as output:
        for directory in directories:
            output.add(directory, arcname=directory.name)


def install_scripted_agent(
    script: str, files: list[Path], environment: dict[str, str]
) -> None:
    """Swap the codex adapter for ``script`` in this process only."""
    import rsi_harness.runtime.production as production
    from rsi_loop.harness.agent.codex import CodexAgent

    CodexAgent.install_cmds = [STUB_LAUNCHER]
    adapter = production.RSILoopAgentAdapter
    extra = dict(environment)
    if files:
        extra["RSI_ACCEPTANCE_FILES"] = str(FILES)

    class ScriptedAgent(adapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            merged = {**dict(prepared.environment), **extra}
            return replace(
                prepared,
                command=("/bin/bash", "-c", script),
                environment=tuple(sorted(merged.items())),
            )

        def run(self, request):
            if files:
                with tempfile.TemporaryDirectory(prefix="rsi-acceptance-") as raw:
                    archive = Path(raw) / "files.tar"
                    bundle(files, archive)
                    self._require_runtime().copy_to(request.container, archive, FILES)
            return super().run(request)

    # ProductionRuntimeServices resolves the adapter class at call time.
    production.RSILoopAgentAdapter = ScriptedAgent


def _pair(value: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return key, item


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: scripted_cli --work-script FILE ... -- run TASK ...")
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="scripted_cli")
    parser.add_argument("--work-script", type=Path, required=True)
    parser.add_argument("--work-files", type=Path, action="append", default=[])
    parser.add_argument("--work-env", type=_pair, action="append", default=[])
    options = parser.parse_args(argv[:split])
    install_scripted_agent(
        options.work_script.read_text(),
        [path.resolve() for path in options.work_files],
        dict(options.work_env),
    )
    from rsi_harness.cli import app

    app(args=argv[split + 1 :], prog_name="rsi-harness")


if __name__ == "__main__":
    main()
