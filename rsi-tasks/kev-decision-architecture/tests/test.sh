#!/bin/bash
set -euo pipefail
cd /
exec /opt/kev-venv/bin/python -I -B /tests/evaluate.py
