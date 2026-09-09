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

"""`python -m rsi_loop.visualizer` — standalone entry for the visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from rsi_loop.visualizer.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="rsi-loop-visualizer")
    parser.add_argument("--runs-dir", default="logs/runs", help="Directory of run folders")
    parser.add_argument("--tasks-dir", default="tasks", help="Directory of task JSONs (for score_direction)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    # tasks-dir provides per-task metadata (score_direction). Degrade gracefully
    # to None when the directory is absent so the visualizer still runs.
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        tasks_dir = None
    app = create_app(runs_dir, tasks_dir=tasks_dir)
    print(f"RSI Loop visualizer serving runs from: {runs_dir.resolve()}")
    print(f"RSI Loop visualizer reading tasks from: {tasks_dir.resolve() if tasks_dir else '(none)'}")
    print(f"Open: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
