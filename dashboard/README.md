# RSI Logs dashboard

A standalone viewer for this repository's `rsi-logs/`, separate from the
OpenRSI Index website. It uses Python's standard library and plain HTML/CSS/JS;
no account, Node installation, or backend service is required.

Hosted dashboard: **https://openrsi-logs.zhuofengli12345.workers.dev/**.

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

## Cloudflare deployment

The dashboard runs independently on the `openrsi-logs` Cloudflare Worker.
Its UI and compressed log pages are hosted as static assets, so the deployed
site does not require a local server. This deployment does not change the
OpenRSI Index website or RSI-Handoff.

After updating the committed logs, rebuild and publish from the repository root:

```bash
python3 dashboard/tools/build.py
uv run dashboard/tools/deploy_cloudflare.py
```

Set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` in the environment,
using a token with Workers Scripts edit access to the account. Alternatively,
pass `--credentials-file /private/path/cloudflare.json`, pointing to a private
JSON file with `api_token` and `account_id` fields. Keep credentials outside
the repository. The account must already have a `workers.dev` subdomain.

The publisher targets only `openrsi-logs` and refuses to overwrite an existing
Worker unless it carries this dashboard's ownership tag. It uploads only the
UI and log pages referenced by the generated index. Use `--prepare-only` to
validate the asset inventory without credentials or network requests.
The generated archive and local `.cloudflare/` deployment receipt are ignored
by Git. Repeat the build and deploy commands to refresh the hosted snapshot.

## Checks

```bash
python3 -m unittest discover -s dashboard/tests -p 'test_logs_data.py' -v
uv run --with httpx python3 -m unittest discover -s dashboard/tests -p 'test_cloudflare_deploy.py' -v
```

Browser checks require Playwright and Chromium, plus a generated archive:

```bash
python3 -m unittest discover -s dashboard/tests -p 'test_logs_browser.py' -v
```
