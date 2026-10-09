"""Spec 7 M9's objective acceptance A1-A8, one scenario per test (opt-in).

Each test runs scripts/operator/sandbox_acceptance.sh as root for one
scenario and requires every row it prints (the A-item and its A5 audit) to
pass. The script is the operator's entry point; this is the same check from
pytest: ``sudo RSI_ACCEPTANCE=1 .venv/bin/python -m pytest -m acceptance
tests/acceptance/test_acceptance.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.acceptance.test_operator_scripts import ACCEPTANCE, SCENARIOS

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_objective_acceptance(as_root, scenario):
    # A short scratch directory: the endpoint socket path must fit 107 bytes.
    # It is kept (logs, watch records, audits) when the scenario fails.
    scratch = Path(tempfile.mkdtemp(prefix="rsi-acc-", dir="/var/tmp"))
    result = subprocess.run(
        [str(ACCEPTANCE), "--only", scenario, "--scratch", str(scratch / "s")],
        capture_output=True,
        text=True,
        timeout=4 * 3600,
    )
    rows_file = scratch / "s" / "rows"
    rows = rows_file.read_text().splitlines() if rows_file.exists() else []
    evidence = f"evidence: {scratch}\n{result.stdout[-4000:]}{result.stderr[-2000:]}"
    assert rows, evidence
    assert all(row.split("|")[1] == "PASS" for row in rows), "\n".join(rows)
    assert result.returncode == 0, evidence
    shutil.rmtree(scratch, ignore_errors=True)
