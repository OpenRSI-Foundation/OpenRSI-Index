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

"""Benchmark metadata — loaded from BENCHMARK.yaml in the tasks directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BENCHMARK_FILENAME = "BENCHMARK.yaml"


@dataclass
class BenchmarkMeta:
    name: str
    base_images: dict[str, dict] = field(default_factory=dict)


def load_benchmark(tasks_dir: Path) -> BenchmarkMeta:
    """Load benchmark metadata from tasks_dir/BENCHMARK.yaml."""
    path = tasks_dir / BENCHMARK_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark metadata not found: {path}\n"
            f"Expected a {BENCHMARK_FILENAME} file in the tasks directory."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkMeta(
        name=data["name"],
        base_images=data.get("base_images", {}),
    )
