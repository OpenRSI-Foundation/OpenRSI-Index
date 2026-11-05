#!/bin/bash
set -euo pipefail
# Restore only candidate surfaces; protected source is never repaired here.
exec /opt/asr-venv/bin/python -I -B /opt/asr-reference/restore.py
