#!/bin/sh
if [ "$(cat /app/answer.txt 2>/dev/null)" = "done" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
