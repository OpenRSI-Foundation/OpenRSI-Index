#!/bin/bash
set -eu

if [ "${RSI_HARNESS_EXPECTED_GPU_UUIDS+x}" != x ]; then
    printf 'missing Engine GPU allocation\n' >&2
    exit 1
fi
expected="$RSI_HARNESS_EXPECTED_GPU_UUIDS"
cp /workspace/work-gpus.txt /logs/verifier/work-gpus.txt
: > /logs/verifier/expected-judge-gpus.txt
: > /logs/verifier/judge-gpus.txt
: > /logs/verifier/judge-gpu-devices.txt
if [ -n "$expected" ]; then
    printf '%s\n' "$expected" | tr ',' '\n' \
        > /logs/verifier/expected-judge-gpus.txt
fi
set +e
nvidia-smi --query-gpu=uuid --format=csv,noheader \
    2> /logs/verifier/judge-nvidia-stderr.txt \
    | sed 's/[[:space:]]*$//' > /logs/verifier/judge-gpus.txt
nvidia_status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$nvidia_status" > /logs/verifier/judge-nvidia-status.txt
for device in /dev/nvidia[0-9]*; do
    [ -e "$device" ] || continue
    basename "$device" >> /logs/verifier/judge-gpu-devices.txt
done
if [ "$nvidia_status" -ne 0 ]; then
    : > /logs/verifier/judge-gpus.txt
fi
visible="$(paste -sd, /logs/verifier/judge-gpus.txt)"

printf 'expected physical GPU UUIDs: %s\n' "$expected"
printf 'Judge-visible physical GPU UUIDs: %s\n' "$visible"
actual_proven=0
if [ "$nvidia_status" -eq 0 ]; then
    actual_proven=1
elif [ -z "$expected" ] && [ ! -s /logs/verifier/judge-gpu-devices.txt ]; then
    actual_proven=1
fi
if [ "$actual_proven" -eq 1 ] && [ "$visible" = "$expected" ]; then
    reward=1
    printf 'Work/Judge allocation is ready for exact comparison\n'
else
    reward=0
    printf 'Judge GPU visibility does not match the caller allocation\n'
fi
printf '{"reward": %s}\n' "$reward" > /logs/verifier/reward.json
