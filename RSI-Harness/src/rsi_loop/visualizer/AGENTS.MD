# Visualizer Development Guide

## Architecture

```
rsi_loop/visualizer/
├── server.py           # FastAPI app, all routes, diff logic
├── scanner.py          # Disk scanner: reads logs/runs/ → Run/Submission models
├── models.py           # Dataclasses: Run, Submission, TestResult
├── markdown.py         # Markdown → HTML renderer (Jinja2 filter)
├── __init__.py
├── __main__.py         # CLI entry: `python -m rsi_loop.visualizer`
├── AGENTS.MD           # This file: dev guide and operator runbook
├── parsers/
│   ├── agent_output.py # Claude Code JSONL → Trajectory/Exchange
│   ├── codex_output.py # Codex plain text → Trajectory
│   └── test_output.py  # Raw test output → per-test blocks
└── templates/
    ├── base.html       # Shell: fonts (Inter/JetBrains Mono), Tailwind, HTMX, CSS
    ├── index.html      # Home: grouped run table with compare checkboxes
    ├── task.html       # Per-task run list
    ├── run_detail.html # Run detail: header + ECharts timeline + submissions + diagnostics
    ├── run_overview.html  # Run overview: all tasks in a run
    ├── compare.html    # Per-task compare: overlaid timelines for selected runs on one task
    ├── compare_runs.html  # Cross-run dashboard: 3-col grid of mini-charts across all shared tasks
    ├── submission_detail.html  # Test results table
    ├── trajectory.html # Full agent conversation + minimap
    ├── _macros.html    # Badge/chip macros (pass_rate_badge, score_badge, etc.)
    ├── _turn.html      # Single exchange rendering (recursive for subagents)
    └── _judger_block.html  # HTMX partial: test output block
```

## Key Design Decisions

### Data Flow
- `scanner.py` re-reads disk on every request (no DB). Docker container status is cached for 5s via a single `docker ps -a` call (replaces per-task `docker inspect`).
- Submission ordering: `evolve_state.json` timestamps > file mtime > seq number.
- Score task detection: `task JSON parser ∈ {score_sum, structured_json}` OR `score_direction` present OR `game_mode`.
- For finalized runs, `final_result.json` is primary; if `best_score=None`, recompute from submissions.
- Agent/model detection for in-progress runs: `_peek_agent()` matches both old format (`Running agent: claude`) and new supervisor format (`Agent supervisor attempt N: claude -p ...`); `_peek_model()` scans multiple lines of `agent_output.txt` for the JSON init event (not just the first line).

### Evolution Timeline (ECharts)
- Step line (`step: "end"`) — score only changes at submissions.
- Milestones = each new-best submission (respects `score_direction` for minimize tasks).
- Agent submissions rendered as circles, auto-eval as triangles (shape differentiation).
- When smoothing is active, step-best line fades (thin + low opacity) and the smooth line becomes the primary visual (thick + solid). Matches W&B convention.
- Pin markers with 2-line labels: `agent-X  score` + analysis summary (20 chars).
- Auto log scale when data spans > 3 orders of magnitude.
- Auto zoom Y-axis to non-zero data range for score tasks.
- Score overflow sentinels (>=1e100, inf, nan) are treated as null in `mv()` to prevent axis squashing.
- `safe_num` Jinja filter ensures Python inf/nan → JS `null` (not literal `inf` which is a JS ReferenceError).
- Tooltip: `appendToBody: true` prevents clipping by chart container.
- Click milestone → detail panel with diff + analysis.

### Smoothing (EMA / SMA)
- Dropdown selector with three modes: Off, EMA (default α=0.6), SMA (default window=30).
- Popover UI: click "Smooth: EMA α=0.60" button to expand slider controls.
- Smoothing is applied **only to milestone points** (kept/new-best), not raw/discarded submissions. This produces a monotonic-ish upper-envelope smooth.
- EMA uses W&B standard formula: `last = α·last + (1-α)·x`, `debias = 1 - α^n`, `smoothed = last / debias`. `last` initialized to 0 (not first value).
- SMA: simple sliding window average over milestone values.
- Consistent across all three chart pages (run_detail, compare, compare_runs).

### Run Overview Lazy Charts
- `/run/{run_id}` lists all tasks in the run as cards but does NOT inline submissions; each mini-chart fetches its own data on demand from `/run/{run_id}/{task}/chart-data` when scrolled into view (`IntersectionObserver`, 600px rootMargin).
- Before this, the page inlined every submission for all tasks; a 49-task / 7103-submission run took >20s server-side. With lazy loading the first paint is sub-second and only visible charts pay the I/O cost.
- ECharts instance is created lazily inside `loadChart`; the loading placeholder div is cleared (`innerHTML = ""`) before `echarts.init`. The page-level `refreshAll` / resize handlers must null-check `c.chart` since unloaded charts have `chart: null`.

### Cross-Run Comparison Dashboard
- Entry: index page checkboxes on **run groups** → "Compare groups (N)" button → `/compare-runs?models=<group>`. The `models=` query name is kept for URL compatibility; the values are run-group prefixes, not API model names.
- **Aggregation unit is `run_group_key(run_id)`** (scanner.py): strips a trailing `-<machine>-<14-digit-timestamp>` so sibling runs of one experiment across machines collapse into one avg@N/ELO group (e.g. `glm51-n35-160-152-2026…` → `glm51-n35-160`), while distinct experiments (`0529-glm51-baseline` vs `0530-glm51-new-env`) stay separate. Earlier versions grouped by raw `r.model` (e.g. `glm-5.1`), which mixed unrelated experiments and silently dropped runs with empty model fields.
- **Two-phase scan** in the `/compare-runs` route: first a cheap `list_runs(include_submissions=False)` learns every run's group/ip, then `list_run_tasks(run_id, include_submissions=True)` is invoked only for the run_ids whose group was selected. Full-tree submission scanning up front used to cost ~260s on a 747-run / 12k-submission tree.
- Layout: header with run-group legend (color dot + agent/model chips + avg rank), rank table (collapsible), then 3-column grid of mini-charts.
- Each mini-chart: per-task overlaid running-best step lines for selected groups, with consistent color per group.
- Y-axis bounds: percentile-based (5th-95th) to handle outliers; auto log scale when range > 10^4.
- Global controls: Auto-eval toggle, Show raw dots, Smooth dropdown, X-axis (submission# / elapsed time), task filter (regex).
- Per-task ranking: rank 1 = best on that task's best metric (respects minimize/maximize). Ties share rank. Tasks where all runs scored 0 are skipped. Overflow sentinels (>=1e100) treated as 0. Avg rank per group shown in header.
- Raw dots: uses ECharts `large: true` + `progressive` + `silent` + `animation: false` for performance with 35 charts × thousands of points.
- Template note: in `compare_runs.html` the JS keeps the field name `data-model` for backward compat — the value is the group prefix, used as both a dict key and a CSS-selector lookup.

### Diff System
- Compares `submission.tar.gz` archives between milestones (not adjacent submissions).
- Safe tar extraction: rejects `../`, absolute paths, symlinks, files > 10MB.
- Filters noise: `__pycache__`, `.cache/`, `.zig-cache`, `node_modules`, etc.
- Diff header shows `Diff: prev_round → cur_round`.
- Lazy-loaded via HTMX `hx-trigger="revealed"` (only when `<details>` is opened).

### Trajectory Page
- Left sidebar shows submission markers with actual round labels (populated from `run.submissions` when stop-hook feedback lacks round info).
- Right minimap: collapsed to 6px by default (opacity 0.35), expands to 42px on hover.

## Common Patterns

### Adding a new section to run_detail
1. Add a `<section class="card overflow-hidden mb-6">` block
2. Header: `<div class="px-5 py-3 border-b border-slate-100">` with `<h2>`
3. Content below, use HTMX for lazy loading if heavy

### Adding a new badge/chip
1. Add macro in `_macros.html`
2. Use `chip` class + Tailwind bg/text color
3. Import in template: `{% from "_macros.html" import new_macro %}`

### HTMX patterns used
- `hx-trigger="revealed"` — load when element becomes visible (inside `<details>`)

- `hx-trigger="intersect once"` — load when scrolled into view
- `hx-swap="innerHTML"` — replace content
- `hx-swap="outerHTML"` — replace the element itself (status transitions)

### Score handling gotchas
- Always use `|safe_num` Jinja filter for score/max_score/peak_score in JS templates (Python inf → JS null).
- `mv()` in JS returns `null` for: missing scores on score tasks, inf/nan, values >=1e100.
- Running-best function skips null values (carries forward previous best).
- Ranking skips tasks where all runs scored 0.

## Known Limitations
- No auth — bind to localhost or put behind reverse proxy.
- Scanner re-reads disk on every request — mitigated by a 10s TTL cache on `list_runs()` (both summary and full variants) and a 5s TTL on the Docker `ps -a` listing.
- `list_run_tasks(run_id)` is **not** cached — it re-scans on every call. Compare/run-overview pages call it on demand, so a cold request still does disk I/O.
- `final_result.json` may be stale/wrong — visualizer trusts it for finalized runs.
- Task alias mismatches (5 tasks have different run dir names vs JSON names).
- Game tasks: `max_score` varies per session, `% of max` is misleading.
- `Rounds` count from `final_result` may differ from actual submission directories.
- Compare dashboard with >5 groups may produce hard-to-read mini-charts (colors overlap).
- **SSHFS-backed runs dir is very slow.** When the per-machine subdirs (`logs/runs/<ip>/`) are sshfs mounts, every `os.stat`/`open()` is a round-trip and scanning a single run can take ~27s on a 49-task tree. The visualizer logic is fine — the bottleneck is FUSE+SSH. Mirror to local disk for any interactive use (see Operator Runbook below).

## Operator Runbook

Run commands from the repository root:

```bash
cd /home/tiger/workspace/SE-bench-harness
```

### Start Visualizer

The `--runs-dir` flag chooses where to read run data from. The repo has three usable layouts:

| Path | Speed | Notes |
|------|-------|-------|
| `logs/runs/` | slow (sshfs) | Live data — each `<ip>/` subdir is an sshfs mount to that machine. Fine for one-off / write paths; painful for interactive browsing. |
| `logs/runs.local/` | **fast (local NVMe)** | Symlink to `/data00/sebench-runs/`. Use for all interactive viewing. Refresh with rsync (see below). |
| `logs/runs/_local_before_sshfs_*/` | local | Historical snapshot, kept for reference. |

Foreground (recommended setup — local mirror):

```bash
uv run python -m sebench.visualizer --runs-dir logs/runs.local --tasks-dir tasks --host 0.0.0.0 --port 8000
```

Background (via tmux so it survives the SSH session):

```bash
tmux new-session -d -s sebench_visualizer \
  '.venv/bin/python -m sebench.visualizer --runs-dir logs/runs.local --host 0.0.0.0 --port 8000 \
   >> /home/tiger/workspace/visualizer.log 2>&1'
```

Check the process:

```bash
ps -eo pid,etime,cmd | grep 'sebench.visualizer' | grep -v grep
ss -ltnp | grep ':8000'
```

Open (note: when curl/Playwright run on this host, requests via `127.0.0.1` skip the corporate proxy that otherwise returns 403 for the LAN address):

```text
http://localhost:8000/
```

### Refresh Local Runs Mirror

`logs/runs.local` is a local copy of every machine's `logs/runs/`. Refresh incrementally with rsync:

```bash
for ip in <host1> <host2> <host3>; do
  rsync -aP -e 'ssh -o BatchMode=yes' \
    user@$ip:/path/to/logs/runs/ \
    /data/sebench-runs/$ip/
done
```

To roll back to live data: restart with `--runs-dir logs/runs` instead of `logs/runs.local`.

### Compare Run Groups

From the index page, check 2+ run groups and click "Compare groups". Or navigate directly with one `models=<group-prefix>` per group:

```text
http://localhost:8000/compare-runs?models=0530-deepseek_v4_pro-new-env&models=0530-glm51-new-env&models=0530-gpt55-new-env
http://localhost:8000/compare-runs?models=glm51-n35-160&models=gpt55-n35-160
```

The legacy `?runs=<run_id>` form is still accepted — the server maps each `run_id` to its group key and proceeds.
