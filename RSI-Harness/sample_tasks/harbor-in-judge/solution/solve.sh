#!/bin/bash
# The baseline for `rsi-harness run --agent oracle`. This sample's Judge
# scores Harbor's own oracle and nop agents, not what Work built, so there is
# nothing to build: check only that the sandbox client the Harness injects is
# on PATH, as an Agent would use it.
set -euo pipefail
rsi-sandbox --help > /dev/null
echo "baseline: nothing to build; rsi-sandbox is available"
