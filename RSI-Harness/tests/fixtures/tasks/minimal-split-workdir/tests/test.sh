#!/bin/bash
set -eu

checkpoint_size="$(stat -c %s /workspace/checkpoint.bin)"
checkpoint_blocks="$(stat -c %b /workspace/checkpoint.bin)"
printf '%s %s\n' "$checkpoint_size" "$checkpoint_blocks" \
    > /logs/verifier/checkpoint-metrics.txt

test "$(cat /workspace/round-state.txt)" = "${ROUND_EXPECTED:?missing round state}"
/usr/local/bin/agent-system-tool > /logs/verifier/system-tool.txt

write_status=0
touch /workspace/judge-write || write_status=$?
printf '%s\n' "$write_status" > /logs/verifier/workdir-write-exit-code.txt
test "$write_status" -ne 0

test ! -e /etc/minimal-split-judge-only
printf 'judge-private\n' > /etc/minimal-split-judge-only
printf '%s\n' "${RSI_HARNESS_EXPECTED_GPU_UUIDS:?missing GPU allocation}" \
    > /logs/verifier/judge-gpus.txt
printf '{"reward": 1, "split_workdir": 1}\n' \
    > /logs/verifier/reward.json
