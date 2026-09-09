#!/usr/bin/env bash
set -euo pipefail

torchrun --nnodes 1 --nproc-per-node 8 /task-tools/judge.py
test "$(find /logs/verifier/judge-ranks -maxdepth 1 -type f | wc -l)" -eq 8
printf '{"reward": 1}\n' > /logs/verifier/reward.json
