#!/bin/bash
set -euo pipefail

cd /workspace
python -I /tests/evaluate.py
