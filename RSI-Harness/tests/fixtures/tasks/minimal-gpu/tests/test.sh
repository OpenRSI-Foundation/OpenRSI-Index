#!/bin/bash
set -eu

answer=""
if [ -f /workspace/answer.txt ]; then
    answer="$(cat /workspace/answer.txt)"
fi
test ! -e /etc/rsi-harness-judge-only.txt
printf 'written only by the Judge\n' > /etc/rsi-harness-judge-only.txt
if [ "$answer" = "42" ]; then
    reward=1
    printf 'answer accepted\n'
else
    reward=0
    printf 'expected the two-character answer 42\n'
fi
printf '{"reward": %s, "answer_length": %s}\n' \
    "$reward" "${#answer}" > /logs/verifier/reward.json
