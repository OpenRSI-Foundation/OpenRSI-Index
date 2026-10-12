# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The pinned version of every Agent CLI and of the node they all run on.

This is the only place an Agent's version is written down: its install
commands and the version its transcript reports are built from it. Upgrade
one with ``scripts/bump_agent.py <agent|node> <version>``, which rewrites the
constant below and runs that Agent's Docker tests; keep each one a plain
``NAME = "X.Y.Z"`` line so the script can find it.

Each value is an exact version, downloaded over HTTPS; npm checks the
integrity of what it installs against the registry.
"""

NODE_VERSION = "22.23.3"

CODEX = "0.159.2"
CLAUDE_CODE = "2.1.285"
QWEN_CODE = "0.25.1"
KIMI_CODE = "2.1.1"
OPENCODE = "1.19.0"
ZCODE = "3.15.1"
PI = "1.1.0"

# Not an Agent: OpenCode's grep tool downloads ripgrep on first use when no
# ``rg`` is on PATH, which a no-network Work cannot do, so its image carries
# this release.
RIPGREP = "15.1.0"
