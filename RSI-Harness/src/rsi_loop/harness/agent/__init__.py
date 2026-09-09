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

"""Pluggable agent abstraction for RSI Loop.

Provides :class:`Agent` (base class with config as class attributes)
and concrete implementations per coding agent.  Select via ``create_agent()``.
"""

from rsi_loop.harness.agent.base import Agent
from rsi_loop.harness.agent.factory import (
    create_agent,
    get_agent_class,
    list_agent_classes,
)

__all__ = [
    "Agent",
    "create_agent",
    "get_agent_class",
    "list_agent_classes",
]
