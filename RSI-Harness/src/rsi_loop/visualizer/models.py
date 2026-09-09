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

"""Dataclasses for run/submission/test results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    name: str
    status: str  # PASSED | FAILED | ERROR | SKIPPED | etc.


@dataclass
class Submission:
    round_label: str  # folder name as-is: "1" (legacy), "agent-3", "auto-7"
    seq: int          # numeric part for ordering within a kind
    kind: str         # "agent" | "auto" | "unknown"
    path: Path
    submitted_at: float = 0.0  # mtime of report.json (unix seconds), fallback 0
    submission_id: str = ""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    score: Optional[float] = None
    score_0_100: Optional[float] = None  # deterministic 0-100 rescale of `score`
    max_score: Optional[float] = None
    peak_score: Optional[float] = None
    timed_out: bool = False
    runtime_seconds: float = 0.0
    tests: list[TestResult] = field(default_factory=list)

    @property
    def round(self) -> int:
        """Back-compat alias used by old templates (sorted-by-seq)."""
        return self.seq

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tests:
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts


@dataclass
class Run:
    run_id: str
    task: str
    path: Path
    agent: str = ""
    model: str = ""
    ip: str = ""  # machine subdir for 3-level layouts; "" for flat 2-level
    best_pass_rate: float = 0.0
    best_score: Optional[float] = None
    best_score_0_100: Optional[float] = None  # rescaled (0-100) of best submission
    best_round: str = ""  # folder-name (e.g. "agent-3", "auto-7"). "" if unknown.
    total_rounds: int = 0
    timed_out: bool = False
    runtime_seconds: float = 0.0
    archive_size_bytes: int = 0
    score_direction: str = "maximize"  # from task JSON: "maximize" or "minimize"
    is_score_task: bool = False        # True when task uses score_sum/structured_json parser
    has_final: bool = False  # False ⇒ run is in-progress / aborted
    aborted: bool = False    # True when no final_result.json AND stale mtime
    infra_error: bool = False  # True when run_agent.log has an ERROR-level line (infra failure)
    created_at: float = 0.0  # unix timestamp; task_dir mtime

    # lazy fields
    submissions: list[Submission] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.run_id}/{self.task}"

    @property
    def is_retry(self) -> bool:
        """Retry runs (run_dir name contains '-retry') are excluded from
        model-level aggregation — they re-run a single task and would
        otherwise be merged into the model's normal sample set."""
        return "-retry" in self.run_id

    @property
    def harness_tag(self) -> str:
        return self.agent or "unknown"

    @property
    def model_display(self) -> str:
        """Non-empty model label for UI. Falls back to '-' when unknown."""
        return self.model or "-"

    @property
    def date_label(self) -> str:
        """YYYY-MM-DD grouping key. '-' if unknown."""
        if not self.created_at:
            return "-"
        from datetime import datetime
        return datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d")

    @property
    def model_provider(self) -> str:
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return self.model or ""

    @property
    def has_auto_eval(self) -> bool:
        return any(s.kind == "auto" for s in self.submissions)

    @property
    def auto_eval_count(self) -> int:
        return sum(1 for s in self.submissions if s.kind == "auto")

    @property
    def has_score(self) -> bool:
        return self.is_score_task

    @property
    def has_max_score(self) -> bool:
        return any(s.max_score is not None and s.max_score > 0 for s in self.submissions)

    @property
    def has_rescaled(self) -> bool:
        """True when a rescaled 0-100 score is available for this run.

        Rescaled scores only exist for tasks whose JSON defines a ``rescale``
        section (and whose raw score was finite); pass-rate tasks never have one.
        In summary scans (no submissions loaded) ``best_score_0_100`` is filled
        directly, so it also counts as "has rescaled".
        """
        return self.best_score_0_100 is not None or any(
            s.score_0_100 is not None for s in self.submissions
        )

    @property
    def has_trajectory(self) -> bool:
        return (self.path / "agent_output.txt").is_file()
