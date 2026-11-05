#!/bin/bash
set -euo pipefail
cd /workspace
exec /opt/sglang/bin/python3 -I /tests/evaluate.py
