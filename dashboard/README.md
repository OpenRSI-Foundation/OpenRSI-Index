# RSI Logs dashboard

A standalone viewer for this repository's `rsi-logs/`, separate from the
OpenRSI Index website. It uses Python's standard library and plain HTML/CSS/JS;
no account, Node installation, or backend service is required.

From the repository root:

```bash
python3 dashboard/serve.py
```

Open **http://127.0.0.1:8000/**. The first start generates the archive index and
compressed transcript pages, which can take a few minutes. Stop with Ctrl+C.
To reuse the existing snapshot or choose another port:

```bash
python3 dashboard/serve.py --skip-build --port 8080
```

The overview groups records by **task**. Expand a task to compare its agent
runs, or use Expand all / Collapse all. Claude and GPT use their corresponding
provider logos. Search tasks/models and filter by track, domain, agent, or
outcome; open a run for its score curve, submissions, trajectory, and artifacts.
Transcript search applies to the displayed page. Tool calls, tool results,
and reasoning expand on click.

Supported archives include Codex terminal output, gzip transcripts, numbered
resume sessions, Claude JSONL, Marin ATIF trajectories, GPIC experiment
journals, and the Qwen historical report. Long entries continue across pages
without truncation. Current browsers decode the small gzip pages using their
built-in `DecompressionStream`; raw transcript HTML remains inert text.

Only committed public `rsi-logs/` files are exported. The footer records the
source commit, and artifact links use that exact revision. Missing final
results are labeled "Archived", without inferring a running process or a
successful outcome. GPIC screening metrics and historical reports are not
combined into Harness score curves. Restart without `--skip-build` after
updating the committed logs to refresh the snapshot.

The generated `dashboard/assets/logs-data/` directory is ignored by Git.
To generate static files without starting a server:

```bash
python3 dashboard/tools/build.py
python3 -m http.server 8000 --directory dashboard
```

These commands do not publish or change any external website.

## Checks

```bash
python3 -m unittest discover -s dashboard/tests -p 'test_logs_data.py' -v
```

Browser checks require Playwright and Chromium, plus a generated archive:

```bash
python3 -m unittest discover -s dashboard/tests -p 'test_logs_browser.py' -v
```
