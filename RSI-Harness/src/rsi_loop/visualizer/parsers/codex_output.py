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

"""Parse OpenAI Codex CLI plain-text output into the shared Trajectory model.

Codex format (line-oriented plain text, not JSON):

    Reading additional input from stdin...
    OpenAI Codex v0.122.0 (research preview)
    --------
    workdir: /home/workspace/minitorch
    model: openai/gpt-5.3-codex
    provider: openrouter
    approval: never
    sandbox: danger-full-access
    reasoning effort: high
    session id: 019db3e5-...
    --------
    user
    <user prompt ...>
    codex
    <assistant text>
    exec
    /bin/bash -lc 'ls -la' in /home/workspace/minitorch
     succeeded in 0ms:
    <output>

    exec
    /bin/bash -lc 'rg --files' in /home/workspace/minitorch
     exited 127 in 0ms:
    <output>

    codex
    <more assistant text>
    ...

Markers appear as bare lines: `user`, `codex`, `exec`.
Each `exec` block: line 1 = command + "in <dir>", line 2 = " succeeded in Xms:" or " exited N in Xms:", then body.
"""

from __future__ import annotations

import re
from pathlib import Path

from rsi_loop.visualizer.parsers.agent_output import (
    Exchange,
    ToolCall,
    Trajectory,
)


# Match "<cmd> in <absolute_dir>" at end of line (first line after "exec")
_EXEC_HEAD_RE = re.compile(r"^(.*?) in (/[^\s]\S*)\s*$")
# Match " succeeded in Xms:" / " exited N in Xms:" (note the leading space)
_EXEC_RESULT_RE = re.compile(r"^ (succeeded|exited \d+) in \d+m?s:$")


def parse(path: Path) -> Trajectory:
    traj = Trajectory()
    if not path.is_file():
        return traj

    text = path.read_text(errors="replace")
    lines = text.splitlines()
    if not lines:
        return traj

    # --- 1. Header (between the first two "--------" lines) ---
    header_meta: dict[str, str] = {}
    i = 0
    while i < len(lines) and lines[i].strip() != "--------":
        ln = lines[i].strip()
        if ln.startswith("OpenAI Codex"):
            header_meta["codex_version"] = ln
        i += 1
    i += 1  # skip opening "--------"
    while i < len(lines) and lines[i].strip() != "--------":
        ln = lines[i]
        if ":" in ln:
            k, _, v = ln.partition(":")
            header_meta[k.strip().replace(" ", "_")] = v.strip()
        i += 1
    i += 1  # skip closing "--------"

    traj.meta = {
        "model": header_meta.get("model", ""),
        "provider": header_meta.get("provider", ""),
        "agent_cli": "codex",
        "session_id": header_meta.get("session_id", ""),
        "workdir": header_meta.get("workdir", ""),
        "sandbox": header_meta.get("sandbox", ""),
        "approval": header_meta.get("approval", ""),
        "reasoning_effort": header_meta.get("reasoning_effort", ""),
        "codex_version": header_meta.get("codex_version", ""),
    }

    # --- 2. Body walk ---
    exchanges: list[Exchange] = []
    next_idx = 0
    current_assistant: Exchange | None = None
    tool_counter = 0

    def add_exchange(role: str, text_body: str = "") -> Exchange:
        nonlocal next_idx
        ex = Exchange(idx=next_idx, role=role, text=text_body.strip())
        next_idx += 1
        exchanges.append(ex)
        return ex

    def consume_block_until_marker() -> str:
        """Return body text starting at current `i`; advance `i` to the marker line."""
        nonlocal i
        buf: list[str] = []
        while i < len(lines):
            lt = lines[i].strip()
            if lt in ("user", "codex", "exec"):
                break
            buf.append(lines[i])
            i += 1
        return "\n".join(buf)

    while i < len(lines):
        marker = lines[i].strip()

        if marker == "user":
            i += 1
            body = consume_block_until_marker()
            add_exchange("user", body)
            current_assistant = None
            continue

        if marker == "codex":
            i += 1
            body = consume_block_until_marker()
            current_assistant = add_exchange("assistant", body)
            continue

        if marker == "exec":
            i += 1
            if i >= len(lines):
                break
            # line 1: "<cmd> in <workdir>"
            head = lines[i]
            m = _EXEC_HEAD_RE.match(head)
            if m:
                cmd = m.group(1).strip()
                workdir = m.group(2).strip()
            else:
                cmd = head.strip()
                workdir = ""
            i += 1
            # line 2: " succeeded in Xms:" or " exited N in Xms:"
            result_header = ""
            if i < len(lines) and _EXEC_RESULT_RE.match(lines[i]):
                result_header = lines[i].strip()
                i += 1
            # body: everything until the next marker
            body = consume_block_until_marker()

            tool_counter += 1
            is_error = "exited" in result_header and "exited 0" not in result_header
            inp: dict = {"command": cmd}
            if workdir:
                inp["workdir"] = workdir
            tool = ToolCall(
                id=f"codex-tool-{tool_counter}",
                name="Bash",
                input=inp,
                result_content=f"{result_header}\n{body}".strip() if result_header else body.rstrip(),
                result_is_error=is_error,
                has_result=True,
            )
            if current_assistant is None:
                current_assistant = add_exchange("assistant", "")
            current_assistant.tool_calls.append(tool)
            continue

        # Unknown line (pre-header noise or stray text) — skip
        i += 1

    traj.exchanges = exchanges
    traj.tool_use_count = sum(len(ex.tool_calls) for ex in exchanges)
    traj.total_events = len(exchanges)
    return traj
