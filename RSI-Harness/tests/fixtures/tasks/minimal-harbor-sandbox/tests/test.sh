#!/bin/sh
set -eu
test "$(cat /workspace/answer.txt)" = "managed-sandbox-ok"
printf '1\n' > /logs/verifier/reward.txt
