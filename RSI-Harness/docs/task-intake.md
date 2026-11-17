# Runtime Task Intake API (`ale.task.intake.v1`)

`rsi-harness serve` lets an upstream task brewer (`ale-brew-science`) push
reviewed, sealed task bundles straight into RSI-Harness. Each accepted task is
staged as a native OpenRSI task and run exactly as OpenRSI-X runs its own:
a plain `rsi-harness run`, with no second review.

The wire contract is `ale.task.intake.v1`: the routes, `X-ALE-*` headers,
HMAC signature, receipts, status views and webhook below. Every runtime that
serves it reports itself in two fields; RSI-Harness answers
`"service": "rsi-intake"` and `"runtime": "rsi"`.

## Start the server

```bash
# Quick integration: Bearer token only, webhook derived from the Brewer URL
RSI_INTAKE_TOKEN=shared-bearer-token \
RSI_INTAKE_WEBHOOK_URL=auto \
rsi-harness serve --host 0.0.0.0 --port 7472

# Production: HMAC + Bearer, a separate webhook secret, one Brewer origin,
# and the run profile every intake task is run with
RSI_INTAKE_SECRET=shared-hmac-dispatch-secret \
RSI_INTAKE_TOKEN=shared-bearer-token \
RSI_INTAKE_WEBHOOK_SECRET=shared-hmac-webhook-secret \
RSI_INTAKE_WEBHOOK_URL=https://brewer.example.org/api/v1/webhooks/ale-status \
RSI_INTAKE_ALLOWED_ORIGINS=https://brewer.example.org \
rsi-harness serve --host 127.0.0.1 --port 7472 --mode run --concurrency 1 \
  --agent codex --model <model> --timeout 3600 \
  --judge-llm-auth api --cluster <profile.toml>
```

The server refuses to start unless `RSI_INTAKE_SECRET` or `RSI_INTAKE_TOKEN`
is set.

| Variable | Brewer (`.env.platform`) counterpart | Meaning |
| --- | --- | --- |
| `RSI_INTAKE_TOKEN` | `ALE_BREW_ALE_DISPATCH_TOKEN` | Bearer token for dispatches and queries |
| `RSI_INTAKE_SECRET` | `ALE_BREW_ALE_DISPATCH_SECRET` | HMAC secret for `X-ALE-Signature` |
| `RSI_INTAKE_WEBHOOK_URL` | (fixed `/api/v1/webhooks/ale-status`) | Default status callback URL, or `auto` (derived from the Brewer's bundle URL); a dispatch's `status_webhook_url` overrides it |
| `RSI_INTAKE_WEBHOOK_SECRET` | `ALE_BREW_ALE_WEBHOOK_SECRET` | Webhook HMAC secret (falls back to the secret, then the token) |
| `RSI_INTAKE_ALLOWED_ORIGINS` | | Comma-separated origins allowed for downloads and webhooks |
| `RSI_INTAKE_STATE_DIR` | | Records (`intake.db`), bundles and attempts (default `.rsi-harness/intake`) |

On the Brewer side set
`ALE_BREW_ALE_DISPATCH_URL=http://<host>:7472/v1/intake/tasks`.

## Routes

| Method & path | Auth | Purpose |
| --- | --- | --- |
| `GET /healthz`, `GET /v1/health` | public | Liveness and protocol version |
| `POST /v1/intake/tasks` | HMAC and/or Bearer | Accept a `pull_url` JSON dispatch or a `multipart_bundle` upload |
| `GET /v1/intake/tasks?status=&limit=` | Bearer | List tasks |
| `GET /v1/intake/tasks/{ref}` | Bearer | One task by id, `dispatch_id` or `idempotency_key` |
| `POST /v1/intake/tasks/{ref}/retry` | Bearer | Re-queue a `failed` task |

Errors are `{"detail": "<reason>"}`: `401` auth, `409` same idempotency key
with a different `artifact.sha256`, `411` a multipart upload without
`Content-Length`, `413` too large (checked before the body is read), `422`
schema, header/body ID mismatch or upload digest mismatch. A first dispatch returns `202`; a
replay of the same key and digest returns `200` with `created: false`.

With both a secret and a token configured, a dispatch must pass both. With
only a token, `X-ALE-Signature` is not checked: a Brewer without a dispatch
secret signs with its own session secret.

## What happens to a task

1. **Fetch**: the uploaded bundle, or `artifact.download_url` (a Bearer token
   is sent only when the URL is not pre-signed), checked against
   `artifact.sha256` and a 512 MiB limit. Up to 5 redirects are followed,
   each hop held to `RSI_INTAKE_ALLOWED_ORIGINS`; `Accept-Encoding: identity`
   keeps the hashed bytes the file the Brewer hashed. The bundle is kept, so
   a retry never needs the (expiring) URL again.
2. **Unpack safely**: `.zip`, `.tar`, `.tar.gz`/`.tgz`, `.tar.zst` (detected
   by content, not name); no absolute paths, `..`, backslashes, links or
   special files; at most 2,000 entries.
3. **Verify the seal** of a Brewer bundle (`artifact.json`,
   `offline-report.json`, `task/`): the report digest, the report's
   `task_digest`, no failed offline checks, and every `task/` entry as
   `{path, kind, mode, size, digest}` in sorted order, where `mode` is the
   permission bits and directories have size 0 and the digest of empty bytes;
   `task_digest` is the SHA-256 of that list as sorted-key JSON. Digests may
   be written `sha256:<hex>` or bare hex, in any case. A plain task folder with
   `task.toml` is accepted unsealed.
4. **Stage as a native task**:
   - `schema_version = "1.4+ale+openrsi"` (any `1.4+…`) becomes `"1.4"`;
   - `rsi.evaluation.max_submissions` → `[metadata.rsi_harness.verifier]`;
     `rsi.runtime.clean_image` → the same table; `score_direction`,
     `cooldown_sec`, `primary_reward` → the run's options;
     `rsi.domain.category/subcategory` → `[metadata]`;
   - `rsi.runtime.sandbox` and `rsi.resources.world` are refused for now;
   - `[rsi]` and `[comment]` are kept under `[metadata.openrsi]`;
   - a missing `[environment].gpus` gets Harbor's default, `0`;
   - `reward.txt` needs no bridge (the Harness reads it after `reward.json`).

   Every change is listed in `result.adaptation_notices`.
5. **Run** in the dispatch's `execution_mode`, or the server's `--mode`:

   | Mode | Does |
   | --- | --- |
   | `lint` | Compile the staged task |
   | `prepare` | Lint, then a `--dry-run` on the configured cluster |
   | `validate` | Lint, then `rsi-harness run --agent oracle` (the task's `solution/` must score) |
   | `run` (default) | Validate, then `rsi-harness run --agent <--agent>`; a failed oracle stops before the Agent |

   Runs use the server's run profile (`--agent`, `--model`, `--cluster`,
   `--gpus`, `--sandbox-policy`, `--judge-llm-*`, `--timeout`). The
   `RSI_INTAKE_*` variables are removed from the run's environment.

## Status and results

`running` is reported when an attempt starts, then `completed` or `failed`
(the webhook enum is exactly `acknowledged | running | completed | failed`).
The webhook is
`POST {dispatch_id, idempotency_key, status, ale_task_id, detail, response_body}`
signed `X-ALE-Signature: sha256=HMAC_SHA256(webhook_secret, raw_body)`, retried
three times; delivery state is in `webhook_delivered_at` /
`webhook_last_error`.

`result` carries the run's own `final_result.json`, the summary every OpenRSI
run produces (`status`, `best_score`, `best_rewards`, `best_round`,
`total_rounds`, `agent`, `model`, `runtime_seconds`, `timed_out`), plus
`run_id`, `exit_code`, `validation` (the oracle run) and the staging fields
(`task_id`, `sealed`, `task_digest`, `adapted`, `adaptation_notices`).

Records live in SQLite; on restart, `acknowledged` and `running` tasks are
queued again.

## Task ids

`ale_task_id`/`task_id` are one canonical id per task:
`[task].name` (or the dispatch's `task_name`) as `org/name`, slugged; a name
without an org takes `openrsi-x` / `openrsi-index` from `task_repo`, else the
first part of `domain`, else `openrsi`.

## Known gaps

- The seal format above records `mode` in each entry, which the interface
  spec's shorter `(path, kind, size, sha256)` summary leaves out; it has not
  yet been checked against a bundle the Brewer produced.
- `rsi.runtime.sandbox` and `rsi.resources.world` have no agreed meaning yet.
