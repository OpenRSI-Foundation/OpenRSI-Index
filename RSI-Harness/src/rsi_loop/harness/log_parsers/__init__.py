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

from rsi_loop.harness.log_parsers.pytest_v import parse_pytest_v
from rsi_loop.harness.log_parsers.score_sum import parse_score_sum
from rsi_loop.harness.log_parsers.structured_json import parse_structured_json

MAP_TASK_TO_PARSER: dict[str, callable] = {
    "pytest_v": parse_pytest_v,
    "score_sum": parse_score_sum,
    "structured_json": parse_structured_json,
}


def get_parser(parser_name: str) -> callable:
    if parser_name not in MAP_TASK_TO_PARSER:
        raise ValueError(
            f"Unknown parser: {parser_name}. Available: {list(MAP_TASK_TO_PARSER.keys())}"
        )
    return MAP_TASK_TO_PARSER[parser_name]
