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

"""Parser for `pytest -v` output.

Verbose pytest output format:
  tests/test_operators.py::test_add PASSED
  tests/test_operators.py::test_mul FAILED
  tests/test_operators.py::test_neg ERROR

With --forked, crashed tests may appear as:
  FAILED                                     [  5%]
  (no test name — subprocess crashed)

Summary line:
  === 25 passed, 3 failed, 1 error in 12.34s ===
  === 30 passed in 8.21s ===
"""

from __future__ import annotations

import re

from rsi_loop.harness.constants import PASSED, FAILED, ERROR


def _parse_summary(test_output: str) -> dict[str, int] | None:
    """Parse the pytest summary line. Returns {passed, failed, error} counts or None."""
    # Match the summary line like: "= 14 failed, 212 passed, 4 xfailed in 361.11s ="
    summary_match = re.search(r"=+\s+([\d\w, ]+)\s+in\s+[\d.]+", test_output)
    if not summary_match:
        return None

    summary_text = summary_match.group(1)
    counts = {"passed": 0, "failed": 0, "error": 0}
    for m in re.finditer(r"(\d+)\s+(\w+)", summary_text):
        count = int(m.group(1))
        label = m.group(2)
        if label in ("passed", "xfailed", "xpassed"):
            counts["passed"] += count
        elif label == "failed":
            counts["failed"] += count
        elif label in ("error", "errors"):
            counts["error"] += count
        # skip "warnings", "skipped", "deselected", etc.
    return counts


def parse_pytest_v(test_output: str) -> list[dict]:
    """
    Parse pytest -v output.

    Returns list of {"name": str, "status": "PASSED"|"FAILED"|"ERROR"}.

    Uses per-test lines for names, but trusts the summary line for total counts.
    This handles pytest-forked crashes where FAILED appears without a test name.
    """
    results: list[dict] = []

    # Match lines like:
    #   tests/test_ops.py::test_add PASSED
    #   tests/test_ops.py::TestClass::test_method FAILED
    #   tests/test_ops.py::test_neg ERROR
    pattern = re.compile(
        r"^([\w/.\-\[\]]+(?:::[\w.\-\[\]]+)+)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)",
        re.MULTILINE,
    )

    named_passed: list[str] = []
    named_failed: list[str] = []
    named_error: list[str] = []

    for match in pattern.finditer(test_output):
        name = match.group(1)
        raw_status = match.group(2)

        if raw_status in ("PASSED", "XFAIL", "XPASS"):
            named_passed.append(name)
        elif raw_status == "FAILED":
            named_failed.append(name)
        elif raw_status == "SKIPPED":
            continue
        else:
            named_error.append(name)

    # Also match collection errors: "ERROR tests/test_xxx.py" (no :: separator)
    collection_pattern = re.compile(
        r"^ERROR\s+([\w/.\-]+\.py)(?:\s|$)", re.MULTILINE
    )
    for match in collection_pattern.finditer(test_output):
        name = match.group(1)
        if name not in named_error:
            named_error.append(name)

    # Try to get authoritative counts from summary line
    summary = _parse_summary(test_output)

    if summary:
        total_passed = summary["passed"]
        total_failed = summary["failed"]
        total_error = summary["error"]
    elif named_passed or named_failed or named_error:
        total_passed = len(named_passed)
        total_failed = len(named_failed)
        total_error = len(named_error)
    else:
        # No test output at all
        return [{"name": "unknown", "status": ERROR}]

    # Emit named results first
    for name in named_passed:
        results.append({"name": name, "status": PASSED})
    for name in named_failed:
        results.append({"name": name, "status": FAILED})
    for name in named_error:
        results.append({"name": name, "status": ERROR})

    # Fill in unnamed entries from summary (crashed forked tests, etc.)
    unnamed_passed = total_passed - len(named_passed)
    unnamed_failed = total_failed - len(named_failed)
    unnamed_error = total_error - len(named_error)

    for i in range(max(0, unnamed_passed)):
        results.append({"name": f"unnamed_passed_{i + 1}", "status": PASSED})
    for i in range(max(0, unnamed_failed)):
        results.append({"name": f"unnamed_crashed_{i + 1}", "status": FAILED})
    for i in range(max(0, unnamed_error)):
        results.append({"name": f"unnamed_error_{i + 1}", "status": ERROR})

    return results
