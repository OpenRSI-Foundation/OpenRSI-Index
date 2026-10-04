#!/bin/bash
set -euo pipefail
cd /
exec /opt/venv/bin/python -B -I /tests/launch.py
