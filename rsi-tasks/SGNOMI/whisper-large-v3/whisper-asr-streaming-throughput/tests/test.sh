#!/bin/bash
set -euo pipefail
cd /
exec /opt/asr-venv/bin/python -I -B /tests/evaluate.py
