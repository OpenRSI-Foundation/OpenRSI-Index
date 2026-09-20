#!/bin/bash
set -euo pipefail
cd /workspace
candidate_name="reward"".json"
source_name="baseline_reward"".json"
install -m 0644 "/opt/peginsert_public/${source_name}" "/workspace/${candidate_name}"
mkdir -p /workspace/research
