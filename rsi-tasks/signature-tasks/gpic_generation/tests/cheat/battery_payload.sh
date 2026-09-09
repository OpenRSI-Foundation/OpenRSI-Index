#!/bin/bash
set -uxo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export GPIC_WORK_ROOT="${GPIC_WORK_ROOT:-$TASK_ROOT}"
pip install --no-cache-dir -q scipy scikit-learn 2>/dev/null
pip install --no-cache-dir -q --no-deps "$GPIC_WORK_ROOT/gpic/gpic_eval"
export TORCH_HOME="$GPIC_WORK_ROOT/torch_home" HF_HUB_OFFLINE=1
export GPIC_QWEN_DIR="$GPIC_WORK_ROOT/hf_models/Qwen3-1.7B"
python "$SCRIPT_DIR/run_battery.py"
