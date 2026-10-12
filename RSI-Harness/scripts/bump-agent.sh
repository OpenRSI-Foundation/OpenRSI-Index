#!/usr/bin/env bash
# Upgrade one pinned Agent CLI (or node) and run its Docker tests; see
# scripts/bump_agent.py.
set -euo pipefail
exec uv run python "$(dirname "$0")/bump_agent.py" "$@"
