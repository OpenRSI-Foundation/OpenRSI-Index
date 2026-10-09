#!/bin/sh
# Reward 1 if /tmp/hello.txt holds exactly "Hello, RSI!" (a trailing newline
# is fine), else 0.
if [ "$(cat /tmp/hello.txt 2>/dev/null)" = "Hello, RSI!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
