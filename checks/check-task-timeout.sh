#!/bin/bash
# Research budgets are task contracts, not GitHub runner limits. No upper cap.
set -euo pipefail
python3 - "$@" <<'PY'
import math
from pathlib import Path
import sys
import tomllib

paths = [Path(arg) / 'task.toml' for arg in sys.argv[1:]] if len(sys.argv) > 1 else sorted(Path('rsi-tasks').rglob('task.toml'))
failed = False
for path in paths:
    try:
        config = tomllib.loads(path.read_text())
        for section in ('agent', 'verifier'):
            value = config.get(section, {}).get('timeout_sec')
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError('timeout must be a positive finite number')
        print(f'PASS {path}: valid task budget (no CI duration cap)')
    except (OSError, ValueError):
        print(f'FAIL {path}: missing/invalid config or non-positive/non-finite timeout')
        failed = True
sys.exit(1 if failed else 0)
PY
