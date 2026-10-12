#!/bin/sh
# Installed as /usr/local/bin/rsi-agent-driver in a driven Agent's Work image.
# The absolute node path is the one the Agent's install commands pin, not
# whatever node the task image puts first on PATH.
exec /usr/local/bin/node /opt/rsi-harness/agent-driver.mjs "$@"
