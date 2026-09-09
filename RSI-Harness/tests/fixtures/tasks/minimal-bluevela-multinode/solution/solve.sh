#!/usr/bin/env bash
set -euo pipefail

torchrun --nnodes 2 --nproc-per-node 8 /task-tools/smoke.py
rsi-submit
