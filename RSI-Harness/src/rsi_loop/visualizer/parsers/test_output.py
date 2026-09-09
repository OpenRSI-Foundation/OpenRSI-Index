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

"""Extract per-test judger output from a raw test log.

Currently tuned for pytest verbose output (the dominant format across tasks):
    =================================== FAILURES ===================================
    _______________________________ test_name _______________________________
    <traceback / captured stdout / etc.>
    _______________________________ test_name_2 _______________________________
    ...

Also handles the `=== ERRORS ===` section the same way. Unknown / non-pytest
formats fall back to "not available".

API:
    cache = TestOutputIndex(submission_dir / "test_output.txt")
    block = cache.block_for("tests/test_autodiff.py::test_chain_rule1")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# "_______________________________ name _______________________________"
_BLOCK_HEADER_RE = re.compile(r"^_{3,}\s+(\S.*?)\s+_{3,}\s*$")
# Section markers like "=========== FAILURES ===========" or "=========== ERRORS ==========="
_SECTION_RE = re.compile(r"^=+\s*([A-Z][A-Z ]+?)\s*=+\s*$")


class TestOutputIndex:
    def __init__(self, path: Path):
        self.path = path
        self._by_shortname: dict[str, str] = {}
        self._raw: Optional[str] = None
        self._parsed = False

    def _ensure_loaded(self) -> None:
        if self._parsed:
            return
        self._parsed = True
        if not self.path.is_file():
            return
        try:
            text = self.path.read_text(errors="replace")
        except OSError:
            return
        self._raw = text
        self._build_index(text)

    def _build_index(self, text: str) -> None:
        lines = text.splitlines()
        i = 0
        n = len(lines)
        in_section = False  # inside FAILURES or ERRORS
        current_name: Optional[str] = None
        current_buf: list[str] = []

        def flush():
            if current_name is not None and current_name not in self._by_shortname:
                self._by_shortname[current_name] = "\n".join(current_buf).rstrip()

        while i < n:
            line = lines[i]
            sec_m = _SECTION_RE.match(line)
            if sec_m:
                name = sec_m.group(1).strip()
                flush()
                current_name = None
                current_buf = []
                in_section = name in ("FAILURES", "ERRORS")
                i += 1
                continue

            if in_section:
                hdr = _BLOCK_HEADER_RE.match(line)
                if hdr:
                    flush()
                    current_name = hdr.group(1).strip()
                    current_buf = [line]
                    i += 1
                    continue
                if current_name is not None:
                    current_buf.append(line)
            i += 1
        flush()

    @staticmethod
    def _shortname(full: str) -> str:
        # pytest id: "tests/foo.py::test_bar[case]" → "test_bar[case]" and "test_bar"
        after = full.rsplit("::", 1)[-1]
        return after

    def block_for(self, full_name: str) -> Optional[str]:
        """Return the judger block for a given test id, or None."""
        self._ensure_loaded()
        if not self._by_shortname:
            return None
        short = self._shortname(full_name)
        if short in self._by_shortname:
            return self._by_shortname[short]
        # pytest strips parametrize brackets in some formats — try base name too.
        base = short.split("[", 1)[0]
        return self._by_shortname.get(base)

    def raw(self, max_bytes: int = 500_000) -> Optional[str]:
        self._ensure_loaded()
        if self._raw is None:
            return None
        if len(self._raw) > max_bytes:
            return self._raw[:max_bytes] + f"\n\n[... truncated, total {len(self._raw)} bytes ...]"
        return self._raw
