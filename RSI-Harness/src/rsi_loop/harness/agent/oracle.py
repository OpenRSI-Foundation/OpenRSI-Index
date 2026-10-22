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

"""The oracle: a task's reference solution in place of a model Agent.

It validates a task's Judge on the real run path. The Harness stages the
task's ``solution/`` at ``/solution`` in Work, owned by the Work user;
``rsi-oracle`` runs ``solve.sh`` in the working directory with the exec user
every Agent gets, removes ``/solution`` (or empties it when that user cannot
unlink it from ``/``), submits through ``rsi-submit`` and exits with ``solve.sh``'s
status. Like Harbor's oracle it submits even after ``solve.sh`` fails, so the
Judge's output is there to read.

No model, no credential and no CLI: the Work image is the task image with the
standard Work layer and the launcher.
"""

from __future__ import annotations

import shlex

from rsi_loop.harness.agent.base import Agent

ORACLE_LAUNCHER = "/usr/local/bin/rsi-oracle"
SOLUTION_DIR = "/solution"
# How many times rsi-oracle submits, set by the Harness.
SUBMISSIONS_ENV = "RSI_ORACLE_SUBMISSIONS"


class OracleAgent(Agent):
    name = "oracle"
    display_name = "Oracle"
    install_cmds: list[str] = []
    run_cmd = "rsi-oracle {cwd} " + SOLUTION_DIR
    # Never read: the Harness passes an oracle no credential.
    api_key_env = ""
    image_files = {ORACLE_LAUNCHER: ("harness/agent/oracle/rsi-oracle", 0o755)}
    runs_task_solution = True

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
        del prompt_path, model, internet, resume, continuation, session_id
        return f"rsi-oracle {shlex.quote(cwd or '.')} {SOLUTION_DIR}"
