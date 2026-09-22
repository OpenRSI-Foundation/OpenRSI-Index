#!/bin/bash
set -euo pipefail
# Source restoration only. The task author explicitly defers baseline training
# and its single score to later Work-side environment validation.
python /opt/kev-reference/restore.py
