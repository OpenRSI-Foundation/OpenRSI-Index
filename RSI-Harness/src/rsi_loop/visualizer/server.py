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

"""FastAPI app for the RSI Loop visualizer.

Mounts:
    GET /                                              → index (runs list)
    GET /task/{task}                                   → per-task runs
    GET /run/{run_id}/{task}                           → run detail
    GET /run/{run_id}/{task}/submission/{n}            → submission detail
    GET /run/{run_id}/{task}/submission/{n}/test       → HTMX partial: judger block
    GET /run/{run_id}/{task}/submission/{n}/raw        → raw test_output.txt (plain)
"""

from __future__ import annotations

import difflib
import json

import tarfile
import tempfile
from html import escape as html_escape
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rsi_loop.visualizer.markdown import render_markdown
from rsi_loop.visualizer.parsers.agent_output import Exchange, get_trajectory
from rsi_loop.visualizer.parsers.test_output import TestOutputIndex
from rsi_loop.visualizer.scanner import RunsIndex, run_group_key

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"


def _json_num(value):
    if value is None:
        return None
    try:
        import math
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _format_duration(seconds: float) -> str:
    if not seconds:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, s = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {s}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m"


def _format_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _build_display_exchanges(run_path: Path, trajectory) -> list[Exchange]:
    """Insert the initial prompt into the rendered trajectory view."""
    display_exchanges: list[Exchange] = []
    if trajectory is None:
        return display_exchanges

    display_exchanges = list(trajectory.exchanges)
    prompt_path = run_path / "agent_prompt.md"
    if prompt_path.is_file() and display_exchanges:
        try:
            prompt_text = prompt_path.read_text(errors="replace")
        except OSError:
            prompt_text = ""
        if prompt_text.strip():
            insert_at = 1 if display_exchanges[0].role == "system" else 0
            max_idx = _max_exchange_idx(display_exchanges)
            prompt_ex = Exchange(
                idx=max_idx + 1,
                role="user",
                text=prompt_text,
            )
            display_exchanges.insert(insert_at, prompt_ex)
    return display_exchanges


def _max_exchange_idx(exchanges: list[Exchange]) -> int:
    max_idx = -1
    for exchange in exchanges:
        max_idx = max(max_idx, exchange.idx)
        for tool_call in exchange.tool_calls:
            max_idx = max(max_idx, _max_exchange_idx(tool_call.sub_exchanges))
    return max_idx


def _load_task_meta(tasks_dir: Optional[Path]) -> dict[str, dict]:
    """Build task_id → {score_direction, selection, is_score_task} from task JSONs."""
    meta: dict[str, dict] = {}
    if not tasks_dir or not tasks_dir.is_dir():
        return meta
    for f in tasks_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        judge = data.get("judge", {})
        is_score_task = (
            judge.get("parser") in ("score_sum", "structured_json")
            or "score_direction" in judge
            or data.get("game_mode", False)
        )
        meta[f.stem] = {
            "score_direction": judge.get("score_direction", "maximize"),
            "selection": judge.get("selection", "score_first" if is_score_task else "pass_rate_first"),
            "is_score_task": is_score_task,
        }
    return meta


def _run_sort_value(r) -> tuple[float, float]:
    """Sort key for runs: (pass_rate, score_component).

    pass_rate is the primary key (universal across all tasks).
    For score tasks, score_component breaks ties (negated for minimize).
    """
    score_val = 0.0
    if r.is_score_task and r.best_score is not None:
        score_val = -r.best_score if r.score_direction == "minimize" else r.best_score
    return (r.best_pass_rate, score_val)


_BINARY_EXTENSIONS = frozenset({
    ".gz", ".tar", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".ico", ".pdf", ".pyc", ".pyo", ".so", ".o", ".a", ".class",
    ".jar", ".whl", ".egg",
})
_IGNORE_PATTERNS = frozenset({
    "__pycache__", ".pyc", ".pyo", ".cache", ".zig-cache",
    "node_modules", ".git", "egg-info", ".config/wesnoth",
    "fontconfig",
})
_MAX_DIFF_LINES_PER_FILE = 500
_MAX_FILES_IN_DIFF = 50


def _compute_submission_diff(
    cur_archive: Path, prev_archive: Path,
) -> list[dict[str, str]]:
    """Extract two submission archives and compute unified diffs."""
    diffs: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        cur_dir = Path(tmpdir) / "cur"
        prev_dir = Path(tmpdir) / "prev"
        cur_dir.mkdir()
        prev_dir.mkdir()

        def _safe_extract(archive: Path, dest: Path) -> bool:
            try:
                with tarfile.open(archive, "r:*") as tf:
                    for member in tf.getmembers():
                        if member.name.startswith("/") or ".." in member.name.split("/"):
                            continue
                        if not (member.isfile() or member.isdir()):
                            continue
                        if member.size > 10 * 1024 * 1024:
                            continue
                        tf.extract(member, dest)
            except (tarfile.TarError, OSError):
                return False
            return True

        if not _safe_extract(cur_archive, cur_dir):
            return [{"filename": "(error)", "diff": "Failed to extract current archive"}]
        if not _safe_extract(prev_archive, prev_dir):
            return [{"filename": "(error)", "diff": "Failed to extract previous archive"}]

        cur_files = {
            str(p.relative_to(cur_dir))
            for p in cur_dir.rglob("*") if p.is_file()
        }
        prev_files = {
            str(p.relative_to(prev_dir))
            for p in prev_dir.rglob("*") if p.is_file()
        }
        all_files = sorted(cur_files | prev_files)

        for fname in all_files[:_MAX_FILES_IN_DIFF]:
            if any(pat in fname for pat in _IGNORE_PATTERNS):
                continue
            if any(fname.endswith(ext) for ext in _BINARY_EXTENSIONS):
                status = ""
                if fname not in prev_files:
                    status = "added"
                elif fname not in cur_files:
                    status = "deleted"
                else:
                    status = "changed"
                diffs.append({"filename": fname, "diff": f"(binary file {status})"})
                continue

            prev_path = prev_dir / fname
            cur_path = cur_dir / fname

            try:
                old_lines = prev_path.read_text(errors="replace").splitlines() if prev_path.is_file() else []
            except OSError:
                old_lines = []
            try:
                new_lines = cur_path.read_text(errors="replace").splitlines() if cur_path.is_file() else []
            except OSError:
                new_lines = []

            diff_lines = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{fname}", tofile=f"b/{fname}",
                lineterm="",
            ))
            if not diff_lines:
                continue
            if len(diff_lines) > _MAX_DIFF_LINES_PER_FILE:
                diff_lines = diff_lines[:_MAX_DIFF_LINES_PER_FILE]
                diff_lines.append(f"\n... truncated ({_MAX_DIFF_LINES_PER_FILE} lines shown)")
            diffs.append({"filename": fname, "diff": "\n".join(diff_lines)})

    return diffs


def _render_diff_html(diffs: list[dict[str, str]]) -> str:
    """Turn diff list into colored HTML."""
    if not diffs:
        return '<div class="text-sm text-slate-400 p-3">No changes detected.</div>'
    parts = []
    for d in diffs:
        fname_esc = html_escape(d["filename"])
        parts.append(f'<div class="diff-file">')
        parts.append(f'<div class="diff-hdr">{fname_esc}</div>')
        lines_html = []
        for line in d["diff"].split("\n"):
            esc = html_escape(line)
            if line.startswith("+") and not line.startswith("+++"):
                lines_html.append(f'<span class="diff-add">{esc}\n</span>')
            elif line.startswith("-") and not line.startswith("---"):
                lines_html.append(f'<span class="diff-del">{esc}\n</span>')
            elif line.startswith("@@"):
                lines_html.append(f'<span class="diff-range">{esc}\n</span>')
            else:
                lines_html.append(f'{esc}\n')
        parts.append(f'<pre class="diff-body">{"".join(lines_html)}</pre>')
        parts.append("</div>")
    return "\n".join(parts)


def create_app(
    runs_dir: Path,
    *,
    tasks_dir: Optional[Path] = None,
) -> FastAPI:
    runs_dir = runs_dir.resolve()
    task_meta = _load_task_meta(tasks_dir)
    app = FastAPI(title="RSI Loop Visualizer")
    app.add_middleware(GZipMiddleware, minimum_size=500)
    index = RunsIndex(runs_dir, task_meta=task_meta)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.globals["format_duration"] = _format_duration
    templates.env.globals["format_pct"] = _format_pct
    templates.env.globals["runs_dir"] = str(runs_dir)
    templates.env.filters["markdown"] = render_markdown

    import math

    def _safe_num(value) -> str:
        """Jinja filter: render numeric value as JSON-safe number or 'null'.

        Python's `inf`, `-inf`, and `nan` are not valid JSON/JavaScript literals.
        Treat them as null so the client sees a missing data point.
        """
        if value is None:
            return "null"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "null"
        if math.isnan(v) or math.isinf(v):
            return "null"
        return repr(v)

    templates.env.filters["safe_num"] = _safe_num

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index_page(request: Request):
        runs = index.list_runs(include_submissions=False)
        runs.sort(key=lambda r: r.created_at, reverse=True)

        by_run: dict[str, list] = {}
        for r in runs:
            by_run.setdefault(r.run_id, []).append(r)

        groups: list[dict] = []
        from datetime import datetime
        for run_id, group_runs in by_run.items():
            group_runs = sorted(group_runs, key=_run_sort_value, reverse=True)
            finalized = [r for r in group_runs if r.has_final and not r.infra_error]
            with_subs = [r for r in group_runs if r.total_rounds > 0 and not r.infra_error]
            best_rate = max((r.best_pass_rate for r in with_subs), default=None)
            avg_rate = (sum(r.best_pass_rate for r in with_subs) / len(with_subs)) if with_subs else None
            done_count = len(finalized)
            solved_count = sum(1 for r in finalized if r.best_pass_rate >= 1.0)
            total_count = len(group_runs)
            total_subs = sum(r.total_rounds for r in group_runs)
            created = min((r.created_at for r in group_runs), default=0.0)
            date_label = datetime.fromtimestamp(created).strftime("%Y-%m-%d") if created else "-"
            agents = sorted({r.agent for r in group_runs if r.agent})
            models = sorted({r.model for r in group_runs if r.model})

            task_rows = [{"run": r} for r in group_runs]

            groups.append({
                "run_id": run_id,
                "task_rows": task_rows,
                "best_rate": best_rate,
                "avg_rate": avg_rate,
                "done_count": done_count,
                "solved_count": solved_count,
                "total_count": total_count,
                "total_subs": total_subs,
                "created_at": created,
                "date_label": date_label,
                "agents": agents,
                "models": models,
            })
        groups.sort(key=lambda g: g["created_at"], reverse=True)

        # Run groups for the "compare" selector. A group is a run_id prefix
        # (machine/timestamp suffix stripped) so sibling runs of one experiment
        # across machines merge into one avg@N unit, while distinct experiments
        # (different date/env) stay separate. model may be empty — group by run.
        group_machines: dict[str, set] = {}
        for r in runs:
            if r.is_retry or r.infra_error:
                continue
            group_machines.setdefault(run_group_key(r.run_id), set()).add(r.ip)
        model_options = [
            {"model": g, "machine_count": len(ips)}
            for g, ips in sorted(group_machines.items())
        ]

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "groups": groups,
                "total_runs": len(runs),
                "model_options": model_options,
                "any_rescaled": any(r.has_rescaled for r in runs),
            },
        )

    @app.get("/task/{task}", response_class=HTMLResponse)
    def task_runs(request: Request, task: str):
        runs = [r for r in index.list_runs() if r.task == task]
        runs = sorted(runs, key=_run_sort_value, reverse=True)
        return templates.TemplateResponse(
            request,
            "task.html",
            {
                "request": request,
                "task": task,
                "runs": runs,
            },
        )

    @app.get("/compare-runs", response_class=HTMLResponse)
    def compare_runs_dashboard(
        request: Request,
        models: list[str] = Query(default=[]),
        runs: list[str] = Query(default=[]),
    ):
        """Compare run groups. The unit is the run-group prefix (a run_id with
        its trailing -<machine>-<timestamp> stripped): sibling runs of the same
        experiment across machines are merged. ELO/ranks are computed client-side
        (see template) so per-task toggles stay live; the server only emits
        per-run pass_rates.

        Accepts ?models=<group> directly, or legacy ?runs=<run_id> (mapped to the
        groups of those runs). The ?models= name is kept for URL compatibility.
        """
        # Two-phase scan: first a cheap summary pass (no submissions) over all
        # runs to learn run_id/group/ip, then a full per-run scan ONLY for the
        # run_ids belonging to the selected groups. Scanning every run's
        # submissions up front cost ~260s on a 747-run / 12k-submission tree.
        summaries = [
            r for r in index.list_runs(include_submissions=False)
            if not r.is_retry and not r.infra_error
        ]

        if models:
            wanted_groups = list(dict.fromkeys(models))
        else:
            wanted_run_ids = set(dict.fromkeys(runs))
            wanted_groups = list(dict.fromkeys(
                run_group_key(r.run_id) for r in summaries if r.run_id in wanted_run_ids
            ))
        wanted_set = set(wanted_groups)

        # run_ids (one per machine) whose group is selected → full scan each.
        selected_run_ids = sorted({
            r.run_id for r in summaries if run_group_key(r.run_id) in wanted_set
        })
        all_runs = []
        for rid in selected_run_ids:
            all_runs.extend(index.list_run_tasks(rid, include_submissions=True))

        # group → task → [Run, ...] (one per machine)
        by_group: dict[str, dict[str, list]] = {}
        for r in all_runs:
            g = run_group_key(r.run_id)
            if g in wanted_set:
                by_group.setdefault(g, {}).setdefault(r.task, []).append(r)

        ordered_models = [g for g in wanted_groups if g in by_group]

        palette = _COMPARE_PALETTE
        model_colors = {m: palette[i % len(palette)] for i, m in enumerate(ordered_models)}

        run_meta = [
            {
                "model": m,
                "agent": next((r.agent for ts in by_group[m].values() for r in ts if r.agent), ""),
                "color": model_colors[m],
                "task_count": len(by_group[m]),
                "machine_count": len({r.ip for ts in by_group[m].values() for r in ts}),
            }
            for m in ordered_models
        ]

        # Tasks present for ≥2 selected groups.
        shared_tasks = sorted(
            t for t in {t for m in ordered_models for t in by_group[m]}
            if sum(1 for m in ordered_models if t in by_group[m]) >= 2
        )

        def _valid(r) -> bool:
            return r.has_final or bool(r.submissions)

        def _time_to_best(r):
            """Elapsed seconds from run start to first reaching its best pass
            rate. Used as the tie-break (earlier wins) when pass rates are
            equal. Relative elapsed (not wall-clock) keeps it fair across
            machines that started at different moments. None ⇒ unknown (treated
            as slowest by the client)."""
            subs = [s for s in r.submissions if s.submitted_at]
            if not subs:
                return None
            t0 = r.created_at or min(s.submitted_at for s in subs)
            reached = [s.submitted_at for s in subs if s.pass_rate >= r.best_pass_rate - 1e-9]
            return (min(reached) - t0) if reached else None

        # Per-task chart payload: keep one line per underlying run, colored by
        # model. `model_runs` carries each valid run's {pr, t} for ELO — pr is
        # the best pass rate, t is time-to-best (earlier breaks pass-rate ties).
        task_blocks: list[dict] = []
        for task in shared_tasks:
            runs_for_task = []
            model_runs: dict[str, list[dict]] = {}
            score_direction = "maximize"
            has_score = False
            has_max_score = False
            has_rescaled = False
            for m in ordered_models:
                for r in by_group[m].get(task, []):
                    if r.score_direction == "minimize":
                        score_direction = "minimize"
                    if r.has_score:
                        has_score = True
                    if r.has_max_score:
                        has_max_score = True
                    if r.has_rescaled:
                        has_rescaled = True
                    if _valid(r):
                        model_runs.setdefault(m, []).append(
                            {"pr": r.best_pass_rate, "t": _time_to_best(r)}
                        )
                    if not r.submissions:
                        continue
                    runs_for_task.append({
                        "run_id": r.run_id,
                        "model": m,
                        "ip": r.ip,
                        "color": model_colors[m],
                        "agent": r.agent,
                        "best_pass_rate": r.best_pass_rate,
                        "best_score": r.best_score,
                        "created_at": r.created_at,
                        "submissions": [
                            {
                                "label": s.round_label, "kind": s.kind,
                                "pass_rate": s.pass_rate, "score": s.score,
                                "score_0_100": s.score_0_100,
                                "max_score": s.max_score, "submitted_at": s.submitted_at,
                                "passed": s.passed, "total": s.total_tests,
                            }
                            for s in r.submissions
                        ],
                    })
            if len(model_runs) < 2:
                continue
            task_blocks.append({
                "task": task,
                "score_direction": score_direction,
                "has_score": has_score,
                "has_max_score": has_max_score,
                "has_rescaled": has_rescaled,
                "runs": runs_for_task,
                "model_runs": model_runs,
            })

        return templates.TemplateResponse(
            request,
            "compare_runs.html",
            {
                "request": request,
                "models": ordered_models,
                "run_meta": run_meta,
                "task_blocks": task_blocks,
                "selected_count": len(ordered_models),
            },
        )

    _COMPARE_PALETTE = [
        "#6366f1",  # indigo
        "#10b981",  # emerald
        "#f59e0b",  # amber
        "#ef4444",  # red
        "#06b6d4",  # cyan
        "#a855f7",  # purple
        "#ec4899",  # pink
        "#84cc16",  # lime
        "#f97316",  # orange
        "#0ea5e9",  # sky
    ]

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    def run_overview(request: Request, run_id: str):
        runs = index.list_run_tasks(run_id, include_submissions=False)
        if not runs:
            raise HTTPException(status_code=404, detail="run not found")

        color = _COMPARE_PALETTE[0]
        task_blocks: list[dict] = []
        for r in runs:
            task_blocks.append({
                "task": r.task,
                "score_direction": r.score_direction,
                "has_score": r.has_score,
                "has_max_score": r.has_max_score,
                "best_pass_rate": r.best_pass_rate,
                "best_score": r.best_score,
                "created_at": r.created_at,
                "color": color,
                "submissions": [],
            })

        rep = max(runs, key=lambda r: r.total_rounds, default=runs[0])
        return templates.TemplateResponse(
            request,
            "run_overview.html",
            {
                "request": request,
                "run_id": run_id,
                "agent": rep.agent,
                "model": rep.model,
                "task_blocks": task_blocks,
                "task_count": len(runs),
            },
        )

    @app.get("/run/{run_id}/{task}/chart-data")
    def run_task_chart_data(run_id: str, task: str):
        r = index.get_run(run_id, task)
        if r is None:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse({
            "task": r.task,
            "score_direction": r.score_direction,
            "has_score": r.has_score,
            "has_max_score": r.has_max_score,
            "has_rescaled": r.has_rescaled,
            "best_pass_rate": r.best_pass_rate,
            "best_score": _json_num(r.best_score),
            "best_score_0_100": _json_num(r.best_score_0_100),
            "created_at": _json_num(r.created_at) or 0,
            "color": _COMPARE_PALETTE[0],
            "submissions": [
                {
                    "label": s.round_label,
                    "kind": s.kind,
                    "pass_rate": s.pass_rate,
                    "score": _json_num(s.score),
                    "score_0_100": _json_num(s.score_0_100),
                    "max_score": _json_num(s.max_score),
                    "submitted_at": _json_num(s.submitted_at) or 0,
                    "passed": s.passed,
                    "total": s.total_tests,
                }
                for s in r.submissions
            ],
        })

    @app.get("/compare/{task}", response_class=HTMLResponse)
    def compare_runs(request: Request, task: str, runs: list[str] = Query(default=[])):
        wanted = list(dict.fromkeys(runs))  # preserve order, dedupe
        all_runs = {r.run_id: r for r in index.list_runs() if r.task == task}
        selected = [all_runs[rid] for rid in wanted if rid in all_runs]
        score_direction = "maximize"
        has_score = False
        has_max_score = False
        has_rescaled = False
        for r in selected:
            if r.score_direction == "minimize":
                score_direction = "minimize"
            if r.has_score:
                has_score = True
            if r.has_max_score:
                has_max_score = True
            if r.has_rescaled:
                has_rescaled = True
        palette = _COMPARE_PALETTE
        colors = [palette[i % len(palette)] for i in range(len(selected))]
        return templates.TemplateResponse(
            request,
            "compare.html",
            {
                "request": request,
                "task": task,
                "runs": selected,
                "colors": colors,
                "score_direction": score_direction,
                "has_score": has_score,
                "has_max_score": has_max_score,
                "has_rescaled": has_rescaled,
            },
        )

    @app.get("/run/{run_id}/{task}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str, task: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        has_trajectory = (run.path / "agent_output.txt").is_file()

        log_tail = ""
        log_path = run.path / "run_agent.log"
        if log_path.is_file():
            try:
                lines = log_path.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-40:])
            except OSError:
                pass

        daemon_tail = ""
        daemon_path = run.path / "auto_eval_daemon.log"
        if daemon_path.is_file():
            try:
                lines = daemon_path.read_text(errors="replace").splitlines()
                daemon_tail = "\n".join(lines[-20:])
            except OSError:
                pass

        has_log = log_path.is_file()
        has_daemon_log = daemon_path.is_file()
        has_prompt = (run.path / "agent_prompt.md").is_file()
        sub_analyses: dict[str, str] = {}
        for sub in run.submissions:
            ap = sub.path / "analysis.md"
            if ap.is_file():
                try:
                    text = ap.read_text(errors="replace").strip()
                    if text:
                        sub_analyses[sub.round_label] = text
                except OSError:
                    pass

        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "request": request,
                "run": run,
                "has_trajectory": has_trajectory,
                "log_tail": log_tail,
                "daemon_tail": daemon_tail,
                "has_log": has_log,
                "has_daemon_log": has_daemon_log,
                "has_prompt": has_prompt,
                "sub_analyses": sub_analyses,
            },
        )

    @app.get("/run/{run_id}/{task}/trajectory", response_class=HTMLResponse)
    def run_trajectory(request: Request, run_id: str, task: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        trajectory = get_trajectory(run.path / "agent_output.txt")
        display_exchanges = _build_display_exchanges(run.path, trajectory)

        marker_submissions: dict[int, str] = {}
        if trajectory and trajectory.markers:
            agent_subs = sorted(
                [s for s in run.submissions if s.kind == "agent"],
                key=lambda s: s.seq,
            )
            for i, m in enumerate(trajectory.markers):
                if i < len(agent_subs):
                    sub = agent_subs[i]
                    marker_submissions[m.anchor_idx] = sub.round_label
                    if not m.round_label:
                        m.round_label = sub.round_label
                    if m.pass_rate is None:
                        m.pass_rate = sub.pass_rate
                    if m.score is None and sub.score is not None:
                        m.score = sub.score
                    if m.passed is None and sub.total_tests:
                        m.passed = sub.passed
                        m.total = sub.total_tests

        return templates.TemplateResponse(
            request,
            "trajectory.html",
            {
                "request": request,
                "run": run,
                "trajectory": trajectory,
                "display_exchanges": display_exchanges,
                "marker_submissions": marker_submissions,
            },
        )

    @app.get("/run/{run_id}/{task}/submission/{round_label}", response_class=HTMLResponse)
    def submission_detail(
        request: Request,
        run_id: str,
        task: str,
        round_label: str,
        status: Optional[str] = Query(None, description="Filter test status"),
    ):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        sub = index.load_submission(run_id, task, round_label)
        if sub is None:
            raise HTTPException(status_code=404, detail="submission not found")
        status_counts = sub.status_counts
        tests = sub.tests
        if status:
            tests = [t for t in tests if t.status == status]
        return templates.TemplateResponse(
            request,
            "submission_detail.html",
            {
                "request": request,
                "run": run,
                "sub": sub,
                "tests": tests,
                "status_counts": status_counts,
                "active_status": status,
            },
        )

    @app.get(
        "/run/{run_id}/{task}/submission/{round_label}/test",
        response_class=HTMLResponse,
    )
    def submission_test_block(
        request: Request,
        run_id: str,
        task: str,
        round_label: str,
        name: str = Query(..., description="Test full id"),
    ):
        sub = index.load_submission(run_id, task, round_label)
        if sub is None:
            raise HTTPException(status_code=404, detail="submission not found")
        toi = TestOutputIndex(sub.path / "test_output.txt")
        block = toi.block_for(name)
        return templates.TemplateResponse(
            request,
            "_judger_block.html",
            {
                "request": request,
                "test_name": name,
                "block": block,
            },
        )

    @app.get("/run/{run_id}/{task}/submission/{round_label}/raw", response_class=PlainTextResponse)
    def submission_raw(run_id: str, task: str, round_label: str):
        sub = index.load_submission(run_id, task, round_label)
        if sub is None:
            raise HTTPException(status_code=404, detail="submission not found")
        toi = TestOutputIndex(sub.path / "test_output.txt")
        return toi.raw() or "(no test_output.txt found)"

    @app.get("/run/{run_id}/{task}/submission/{round_label}/archive")
    def submission_archive(run_id: str, task: str, round_label: str):
        """Download this submission's submission.tar.gz (the code patch as
        sent to the judge)."""
        sub = index.load_submission(run_id, task, round_label)
        if sub is None:
            raise HTTPException(status_code=404, detail="submission not found")
        path = sub.path / "submission.tar.gz"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="submission.tar.gz not found")
        return FileResponse(
            path,
            media_type="application/gzip",
            filename=f"{run_id}__{task}__{round_label}.tar.gz",
        )

    @app.get("/run/{run_id}/{task}/log", response_class=PlainTextResponse)
    def run_log(run_id: str, task: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        log_path = run.path / "run_agent.log"
        if not log_path.is_file():
            return "(no run_agent.log found)"
        try:
            return log_path.read_text(errors="replace")
        except OSError:
            return "(error reading log)"

    @app.get("/run/{run_id}/{task}/daemon-log", response_class=PlainTextResponse)
    def daemon_log(run_id: str, task: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        log_path = run.path / "auto_eval_daemon.log"
        if not log_path.is_file():
            return "(no auto_eval_daemon.log found)"
        try:
            return log_path.read_text(errors="replace")
        except OSError:
            return "(error reading log)"

    @app.get("/run/{run_id}/{task}/download")
    def task_download(run_id: str, task: str):
        """Stream the entire task dir (logs + submissions + final archive) as
        a gzipped tar. Typical size 2-5 MB compressed."""
        run = index.get_run(run_id, task)
        if run is None or not run.path.is_dir():
            raise HTTPException(status_code=404, detail="run not found")

        def stream():
            import io
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(str(run.path), arcname=f"{run_id}__{task}")
            buf.seek(0)
            while chunk := buf.read(64 * 1024):
                yield chunk

        return StreamingResponse(
            stream(),
            media_type="application/gzip",
            headers={"Content-Disposition": f'attachment; filename="{run_id}__{task}.tar.gz"'},
        )

    @app.get("/run/{run_id}/{task}/agent-output")
    def agent_output_download(run_id: str, task: str):
        """Download raw agent_output.txt (the agent's full conversation log)."""
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        path = run.path / "agent_output.txt"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="agent_output.txt not found")
        return FileResponse(
            path,
            media_type="text/plain; charset=utf-8",
            filename=f"{run_id}__{task}__agent_output.txt",
        )

    @app.get("/run/{run_id}/{task}/prompt", response_class=PlainTextResponse)
    def agent_prompt(run_id: str, task: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        prompt_path = run.path / "agent_prompt.md"
        if not prompt_path.is_file():
            return "(no agent_prompt.md found)"
        try:
            return prompt_path.read_text(errors="replace")
        except OSError:
            return "(error reading prompt)"

    @app.get("/run/{run_id}/{task}/trace", response_class=PlainTextResponse)
    def raw_trace(run_id: str, task: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        trace_path = run.path / "agent_output.txt"
        if not trace_path.is_file():
            return "(no agent_output.txt found)"
        try:
            return trace_path.read_text(errors="replace")
        except OSError:
            return "(error reading trace)"

    @app.get("/run/{run_id}/{task}/trace/json")
    def trace_json(run_id: str, task: str):
        """Return parsed trajectory as JSON for programmatic analysis."""
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        trace_path = run.path / "agent_output.txt"
        if not trace_path.is_file():
            raise HTTPException(status_code=404, detail="no agent_output.txt")
        events = []
        try:
            with trace_path.open("r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            raise HTTPException(status_code=500, detail="error reading trace")
        return {"task": task, "run_id": run_id, "total_events": len(events), "events": events}

    # ── Submission diff ─────────────────────────────────────────

    @app.get(
        "/run/{run_id}/{task}/submission/{round_label}/diff",
        response_class=HTMLResponse,
    )
    def submission_diff(run_id: str, task: str, round_label: str):
        run = index.get_run(run_id, task)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        subs = sorted(run.submissions, key=lambda s: (s.submitted_at, s.seq))
        cur_sub = next((s for s in subs if s.round_label == round_label), None)
        if cur_sub is None:
            raise HTTPException(status_code=404, detail="submission not found")

        cur_archive = cur_sub.path / "submission.tar.gz"
        if not cur_archive.is_file():
            return HTMLResponse(
                '<div class="text-sm text-slate-400 p-3">No submission.tar.gz found.</div>'
            )

        cur_idx = next(i for i, s in enumerate(subs) if s.round_label == round_label)
        if cur_idx == 0:
            return HTMLResponse(
                '<div class="text-sm text-slate-400 p-3">First submission — no previous version to diff against.</div>'
            )
        prev_sub = subs[cur_idx - 1]
        prev_archive = prev_sub.path / "submission.tar.gz"
        if not prev_archive.is_file():
            return HTMLResponse(
                '<div class="text-sm text-slate-400 p-3">Previous submission archive not found.</div>'
            )

        diffs = _compute_submission_diff(cur_archive, prev_archive)
        header = f'<div class="text-[11px] text-slate-400 mb-2">Diff: {html_escape(prev_sub.round_label)} → {html_escape(round_label)}</div>'
        return HTMLResponse(header + _render_diff_html(diffs))

    return app
