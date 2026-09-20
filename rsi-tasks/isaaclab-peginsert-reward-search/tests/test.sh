#!/bin/bash
set -euo pipefail
cd /workspace
exec python3 -I /tests/evaluate.py
