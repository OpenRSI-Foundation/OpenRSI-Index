#!/bin/sh
# Installed as /usr/local/bin/zcode in the zcode Agent's Work image. ZCode
# ships as a Node bundle; the absolute node path is the one the Agent's
# install commands pin, not whatever node the task image puts first on PATH.
exec /usr/local/bin/node /opt/zcode/zcode.cjs "$@"
