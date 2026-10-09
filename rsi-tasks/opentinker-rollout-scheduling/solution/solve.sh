#!/bin/bash
set -euo pipefail
mkdir -p /workspace/candidate
rm -f /workspace/candidate/policy.py /workspace/candidate/source_manifest.json
cp /opt/opentinker_task/reference/policy.py /workspace/candidate/policy.py
cp /opt/opentinker_task/reference/source_manifest.json /workspace/candidate/source_manifest.json
